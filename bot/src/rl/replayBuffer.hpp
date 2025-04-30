#ifndef RL_REPLAY_BUFFER_HPP_
#define RL_REPLAY_BUFFER_HPP_

#include <absl/log/log.h>

#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>

namespace rl {

template<typename TransitionType>
class ReplayBuffer {
private:
  // --- Internal Types ---
  // LeafIndexType is the index type for leaves within our sum-tree (0 to capacity-1).
  using LeafIndexType = size_t;

public:
  // --- Public Types ---
  using TransitionId = LeafIndexType;

  // --- Configuration ---
  // alpha: How much prioritization to use (0=uniform, 1=full priority)
  // beta: Initial importance sampling exponent (annealed externally, or stays constant)
  // epsilon: Small value added to priorities to ensure non-zero probability
  ReplayBuffer(size_t capacity, float alpha, float beta, float epsilon);

  // Adds a transition and updates PER structures. Buffer must not be full. It is the user's responsibility to delete transitions if the buffer is full before adding new ones.
  TransitionId addTransition(const TransitionType &transition);

  // Deletes a transition from the buffer.
  void deleteTransition(TransitionId id);

  // Result structure for sampling
  struct SampleResult {
    TransitionId transitionId{0xBFBFBFBF};
    float weight;
  };

  // Samples a batch of transitions based on priority.
  std::vector<SampleResult> sample(int count, std::mt19937 &rng);

  // Updates the priorities of specific transitions after they've been processed.
  void updatePriorities(const std::vector<TransitionId> &ids, const std::vector<float> &priorities);

  size_t size() const;
  size_t capacity() const { return capacity_; }

private:
  // --- Configuration ---
  const size_t capacity_;
  const float alpha_;
  const float beta_;
  const float epsilon_;

  // --- PER Data Structures ---
  mutable std::mutex replayBufferMutex_;
  std::vector<float> sumTree_; // Stores priorities p^alpha (size 2*capacity - 1)

  // --- State ---
  // Initially, items are inserted sequentially. However, once the user starts deleting items, we switch to a queue for which position to insert next.
  std::optional<LeafIndexType> nextLeafIndex_{0}; // Next leaf index to write priority to
  std::deque<LeafIndexType> nextLeafIndices_;     // Queue of free indices
  size_t currentBufferSize_{0};                   // Number of valid entries in the buffer/tree
  float maxPriority_{1.0f};                       // Max priority seen so far (p^alpha)

  // --- Helper Methods ---

  // Converts a a leaf index in the sum-tree to a transition ID.
  TransitionId leafIndexToTransitionId(LeafIndexType leafIndex) const { return leafIndex; }

  // Converts a transition ID to a leaf index in the sum-tree.
  LeafIndexType transitionIdToLeafIndex(TransitionId id) const { return id; }

  // Gets the next leaf index to write to.
  LeafIndexType getNextLeafIndex() const;

  // Advances the next leaf index to write to.
  void advanceNextLeafIndex();

  // Calculates priority p^alpha = (|priority| + epsilon)^alpha
  float calculatePriority(float priority) const;

  // Propagates priority change up the sum-tree from a given *tree* index.
  void propagateUpward(size_t treeIndex);

  // Updates the priority at a specific *leaf* index in the sum-tree.
  void updateTreeForChangedLeaf(LeafIndexType leafIndex, float priorityAlpha);

  // Retrieves the leaf index (0 to capacity-1) and priority corresponding to a sampled value.
  std::pair<LeafIndexType, float> retrieveLeaf(float valueToFind) const;

  // Gets the total priority sum (value at the root node).
  float getTotalPriority() const;

  size_t internalSize() const { return currentBufferSize_; }
};

#include "replayBuffer.inl"

} // namespace rl

#endif // RL_REPLAY_BUFFER_HPP_