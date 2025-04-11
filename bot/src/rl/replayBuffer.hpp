#ifndef RL_REPLAY_BUFFER_HPP_
#define RL_REPLAY_BUFFER_HPP_

#include "common/pvpDescriptor.hpp"
#include "common/random.hpp"
#include "rl/observation.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <absl/hash/hash.h>

#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>

namespace rl {

// One data structure holds the raw Observations and Actions. These are stored sequentially for each fight for each character. We do this because building the observations for the model will eventually require frame stacking. Rather than storing the frame-stacked observations in the replay buffer, we store the raw observations and build the frame-stacked observations on the fly just before calling the model.

// Our ReplayBuffer interface:
//  1. Add singular observations
//  2. Sample a batch of transitions
//  3. Update priorities of transitions

// Note: This class does not store any rewards. Those are calculated externally. This allows us to use reward shaping.
class ObservationAndActionStorage {
public:
  struct Index {
    common::PvpDescriptor::PvpId pvpId{};
    sro::scalar_types::EntityGlobalId observerGlobalId{};
    size_t actionIndex{};

    bool havePrevious() const {
      return actionIndex > 0;
    }

    Index previous() const {
      Index result{*this};
      if (actionIndex == 0) {
        throw std::runtime_error("Cannot construct a previous index");
      }
      --result.actionIndex;
      return result;
    }

    bool operator==(const Index& other) const {
      return pvpId == other.pvpId &&
             observerGlobalId == other.observerGlobalId &&
             actionIndex == other.actionIndex;
    }

    // Abseil hashing support
    template <typename H>
    friend H AbslHashValue(H h, const Index& idx) {
      return H::combine(std::move(h), idx.pvpId, idx.observerGlobalId, idx.actionIndex);
    }
  };
  struct ObservationAndActionType {
    Observation observation;
    std::optional<int> actionIndex;
  };

  ObservationAndActionStorage(size_t capacity);
  Index addObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const Observation &observation, std::optional<int> actionIndex);
  ObservationAndActionType getObservationAndAction(Index index) const;

private:
  const size_t capacity_;
  std::map<common::PvpDescriptor::PvpId, std::map<sro::scalar_types::EntityGlobalId, std::deque<ObservationAndActionType>>> buffer_;
};

class ReplayBuffer {
private:
  // --- Internal Types ---
  // LeafIndexType is the index type for leaves within our sum-tree (0 to capacity-1).
  using LeafIndexType = size_t;

  using TransitionType = std::pair<ObservationAndActionStorage::ObservationAndActionType,
                                   ObservationAndActionStorage::ObservationAndActionType>;

public:
  // --- Configuration ---
  // alpha: How much prioritization to use (0=uniform, 1=full priority)
  // beta: Initial importance sampling exponent (annealed externally, or stays constant)
  // epsilon: Small value added to priorities to ensure non-zero probability
  ReplayBuffer(size_t capacity, size_t samplingBatchSize, float alpha, float beta, float epsilon);

  // StorageIndexType is the index type defined by the storage class.
  using StorageIndexType = typename ObservationAndActionStorage::Index;

  // Adds a transition to the internal storage and updates PER structures.
  StorageIndexType addObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const Observation &observation, std::optional<int> actionIndex);

  ObservationAndActionStorage::ObservationAndActionType getObservationAndAction(StorageIndexType index) const;

  // Result structure for sampling
  struct SampleResult {
    StorageIndexType storageIndex; // Used for updating priorities
    TransitionType transition;
    float weight;
  };

  // Samples a batch of transitions based on priority.
  std::vector<SampleResult> sample();

  // Updates the priorities of specific transitions after they've been processed.
  // Uses the indices provided by the storage class.
  void updatePriorities(const std::vector<StorageIndexType>& storageIndices, const std::vector<float>& tdErrors);

  size_t size() const;
  size_t samplingBatchSize() const;

private:
  mutable std::mutex replayBufferMutex_;
  // --- Configuration ---
  const size_t capacity_;
  const size_t samplingBatchSize_;
  const float alpha_;
  const float beta_;
  const float epsilon_;

  // --- Internal Storage ---
  ObservationAndActionStorage storage_;

  // --- PER Data Structures ---
  std::vector<float> sumTree_; // Stores priorities p^alpha (size 2*capacity - 1)
  // Mappings between storage index and sum-tree leaf index
  std::vector<StorageIndexType> leafToStorageMap_; // leafIndex -> storageIndex
  std::unordered_map<StorageIndexType, LeafIndexType, absl::Hash<StorageIndexType>> storageToLeafMap_; // storageIndex -> leafIndex

  // --- State ---
  LeafIndexType currentLeafIndex_{0};              // Next *leaf* index to write priority to (circular)
  size_t currentBufferSize_{0};                    // Number of valid entries in the buffer/tree
  float maxPriority_{1.0f};                        // Max priority seen so far (p^alpha)
  std::mt19937 rng_{common::createRandomEngine()}; // Random number generator

  // --- Helper Methods ---

  // Calculates priority p^alpha = (|tdError| + epsilon)^alpha
  float calculatePriority(float tdError) const;

  // Propagates priority change up the sum-tree from a given *tree* index.
  void propagate(size_t tree_index);

  // Updates the priority at a specific *leaf* index in the sum-tree.
  void updateTree(LeafIndexType leafIndex, float priorityAlpha);

  // Retrieves the leaf index (0 to capacity-1) and priority corresponding to a sampled value.
  std::pair<LeafIndexType, float> retrieveLeaf(float valueToFind) const;

  // Gets the total priority sum (value at the root node).
  float getTotalPriority() const;
};

} // namespace rl

#endif // RL_REPLAY_BUFFER_HPP_