#ifndef RL_OBSERVATION_AND_ACTION_STORAGE_HPP_
#define RL_OBSERVATION_AND_ACTION_STORAGE_HPP_

#include "common/pvpDescriptor.hpp"
#include "common/random.hpp"
#include "rl/observation.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <absl/strings/str_format.h>

#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace rl {

// This data structure holds the raw Observations and Actions. These are stored sequentially for each fight for each character. We do this because building the observations for the model will eventually require frame stacking. Rather than storing the frame-stacked observations, we store the raw observations and build the frame-stacked observations on the fly just before calling the model.
// Note: This class does not store any rewards. Those are calculated externally. This allows us to use reward shaping.
class ObservationAndActionStorage {
public:
  using Id = size_t;
  struct ObservationAndActionType {
    Observation observation;
    std::optional<int> actionIndex;
  };

  ObservationAndActionStorage(size_t capacity);

  // Adds an observation and action to the storage. Returns the ID of the observation (first of pair).
  // If the storage is full, it will delete the oldest PVP and return the IDs of all observations and actions in the deleted PVP (second of pair).
  std::pair<Id, std::vector<Id>> addObservationAndAction(common::PvpDescriptor::PvpId pvpId, const std::string &intelligenceName, const Observation &observation, std::optional<int> actionIndex);

  // Note that insertion & deletion do not invalidate references.
  const ObservationAndActionType& getObservationAndAction(Id id) const;
  bool hasPrevious(Id id) const;

  // Throws an exception if there is no previous observation.
  Id getPreviousId(Id id) const;

private:
  const size_t capacity_;
  mutable std::mutex mutex_;
  size_t nextId_{0};
  std::map<common::PvpDescriptor::PvpId, std::map<std::string, std::deque<ObservationAndActionType>>> buffer_;

  struct BufferIndices {
    BufferIndices() = default;
    BufferIndices(common::PvpDescriptor::PvpId pvpId, const std::string &intelligenceName, size_t actionIndex) : pvpId(pvpId), intelligenceName(intelligenceName), actionIndex(actionIndex) {}
    common::PvpDescriptor::PvpId pvpId;
    std::string intelligenceName;
    size_t actionIndex;

    bool operator==(const BufferIndices &other) const {
      return pvpId == other.pvpId && intelligenceName == other.intelligenceName && actionIndex == other.actionIndex;
    }

    // Abseil hashing support
    template <typename H>
    friend H AbslHashValue(H h, const BufferIndices& idx) {
      return H::combine(std::move(h), idx.pvpId, idx.intelligenceName, idx.actionIndex);
    }
  };

  absl::flat_hash_map<Id, BufferIndices> idToIndexMap_;
  absl::flat_hash_map<BufferIndices, Id> indexToIdMap_;
  std::deque<std::pair<common::PvpDescriptor::PvpId, std::string>> pvpIdAndIntelligenceNameDeque_;

  size_t size() const;
};

} // namespace rl

#endif // RL_OBSERVATION_AND_ACTION_STORAGE_HPP_