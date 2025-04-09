#include "rl/replayBuffer.hpp"

#include <absl/log/log.h>

namespace rl {

ObservationAndActionStorage::ObservationAndActionStorage(size_t capacity) : capacity_(capacity) {}

ObservationAndActionStorage::Index ObservationAndActionStorage::addObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const Observation &observation, std::optional<int> actionIndex) {
  std::deque<ObservationAndActionType> &observations = buffer_[pvpId][observerGlobalId];
  observations.push_back({observation, actionIndex});
  size_t index = observations.size() - 1;
  return {pvpId, observerGlobalId, index};
}

ObservationAndActionStorage::ObservationAndActionType ObservationAndActionStorage::getObservationAndAction(Index index) const {
  auto it = buffer_.find(index.pvpId);
  if (it == buffer_.end()) {
    throw std::out_of_range("Index not found in buffer.");
  }
  auto it2 = it->second.find(index.observerGlobalId);
  if (it2 == it->second.end()) {
    throw std::out_of_range("Index not found in buffer.");
  }
  const auto& observations = it2->second;
  if (index.actionIndex >= observations.size()) {
    throw std::out_of_range("Index out of range in transition vector.");
  }
  return observations.at(index.actionIndex);
}

ReplayBuffer::ReplayBuffer(size_t capacity, size_t samplingBatchSize,
                           float alpha, float beta, float epsilon)
      : capacity_(capacity),
        samplingBatchSize_(samplingBatchSize),
        alpha_(alpha),
        beta_(beta),
        epsilon_(epsilon),
        storage_(capacity),
        sumTree_(2 * capacity - 1, 0.0f),
        leafToStorageMap_(capacity)
{
  if (capacity == 0) {
    throw std::runtime_error("Capacity must be positive.");
  }
  storageToLeafMap_.reserve(capacity);
}

void ReplayBuffer::addObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const Observation &observation, std::optional<int> actionIndex) {
  std::unique_lock lock{replayBufferMutex_};

  // TODO: Remove
  if (currentBufferSize_ == capacity_) {
    // Do not add if buffer is full, for now.
    return;
  }

  // 1. Add transition to internal storage, get its index within that storage
  StorageIndexType storageIndex = storage_.addObservationAndAction(pvpId, observerGlobalId, observation, actionIndex);
  if (storageIndex.actionIndex == 0) {
    // This is the first observation of this pvp for this player. We do not yet have a full transition. Nothing to do.
    return;
  }

  // 2. Determine the leaf index in our sum-tree where this priority will go
  LeafIndexType leafIndex = currentLeafIndex_;

  // 3. Handle potential overwrite in mapping
  if (currentBufferSize_ == capacity_) {
    throw std::runtime_error("Cannot yet handle when buffer reaches capacity.");
    // If buffer is full, the current leafIndex is overwriting an old entry.
    // We need to remove the old storageIndex -> leafIndex mapping.
    StorageIndexType old_storage_idx = leafToStorageMap_[leafIndex];
    storageToLeafMap_.erase(old_storage_idx);
  }

  // 4. Update mappings
  leafToStorageMap_[leafIndex] = storageIndex;
  storageToLeafMap_[storageIndex] = leafIndex;

  // 5. Update the sum-tree using the max priority for the new transition. This is suggested by the paper so that new transitions are always sampled at least once.
  updateTree(leafIndex, maxPriority_);

  // 6. Advance leaf index (circular) and update buffer size
  currentLeafIndex_ = (currentLeafIndex_ + 1) % capacity_;
  if (currentBufferSize_ < capacity_) {
    currentBufferSize_++;
  }
  if (currentBufferSize_%1000 == 0) {
    LOG(INFO) << "Added transition #" << currentBufferSize_ << " to replay buffer.";
  }
}

std::vector<ReplayBuffer::SampleResult> ReplayBuffer::sample() {
  std::unique_lock lock{replayBufferMutex_};
  if (currentBufferSize_ < samplingBatchSize_) {
    throw std::runtime_error("Not enough transitions in buffer to sample a full batch.");
  }

  std::vector<SampleResult> results;
  results.reserve(samplingBatchSize_);

  float total_priority = getTotalPriority();
  if (total_priority <= 0.0f) {
    throw std::runtime_error("Total priority is zero or negative, cannot sample.");
  }

  float segment_size = total_priority / static_cast<float>(samplingBatchSize_);
  float max_weight = 0.0f;

  for (size_t i = 0; i < samplingBatchSize_; ++i) {
    std::uniform_real_distribution<float> dist(i * segment_size, (i + 1) * segment_size);
    float value_to_find = dist(rng_);

    // Retrieve leaf index (within sum-tree) and its stored priority (p^alpha)
    auto [leafIndex, priority_alpha] = retrieve_leaf(value_to_find);

    // Get the corresponding storage index using the map
    const StorageIndexType observation1StorageIndex = leafToStorageMap_[leafIndex];

    // Calculate sampling probability P(i) = p_i^alpha / total_p_alpha
    float sampling_probability = priority_alpha / total_priority;
    if (sampling_probability <= 0.0f) {
      sampling_probability = std::numeric_limits<float>::epsilon(); // Avoid issues
    }

    // Calculate Importance Sampling (IS) weight: w_i = (N * P(i))^-beta
    // Note: Using currentBufferSize_ as N
    float weight = std::pow(static_cast<float>(currentBufferSize_) * sampling_probability, -beta_);
    max_weight = std::max(max_weight, weight); // Track max weight for normalization

    // Create a new SampleResult object with one-lower index
    const StorageIndexType observation0StorageIndex = observation1StorageIndex.previous();

    results.push_back({TransitionType{storage_.getObservationAndAction(observation0StorageIndex), storage_.getObservationAndAction(observation1StorageIndex)}, weight});
  }

  // Normalize weights: w_i = w_i / max_w
  if (max_weight > 0) {
    for (SampleResult &sample : results) {
      sample.weight /= max_weight;
    }
  }

  return results;
}

void ReplayBuffer::updatePriorities(const std::vector<StorageIndexType>& storageIndices, const std::vector<float>& td_errors) {
  std::unique_lock lock{replayBufferMutex_};
  if (storageIndices.size() != td_errors.size()) {
    throw std::runtime_error("Indices and TD errors size mismatch in updatePriorities.");
  }

  for (size_t i = 0; i < storageIndices.size(); ++i) {
    StorageIndexType storageIndex = storageIndices[i];
    auto it = storageToLeafMap_.find(storageIndex);

    if (it != storageToLeafMap_.end()) {
      LeafIndexType leafIndex = it->second; // Found the leaf index in the tree
      float priority_alpha = calculate_priority(td_errors[i]);
      maxPriority_ = std::max(maxPriority_, priority_alpha); // Ensure max_priority stays current
      updateTree(leafIndex, priority_alpha);
    } else {
      // Handle cases where the storage index is not found in the map
      // (e.g., it was overwritten and added again before update)
      // Depending on requirements, could log a warning or ignore.
      // std::cerr << "Warning: Storage index " << storageIndex << " not found for priority update." << std::endl;
    }
  }
}

size_t ReplayBuffer::size() const {
  std::unique_lock lock{replayBufferMutex_};
  return currentBufferSize_;
}

size_t ReplayBuffer::samplingBatchSize() const {
  std::unique_lock lock{replayBufferMutex_};
  return samplingBatchSize_;
}

float ReplayBuffer::calculate_priority(float tdError) const {
  return std::pow(std::abs(tdError) + epsilon_, alpha_);
}

void ReplayBuffer::propagate(size_t tree_index) {
  // Start from parent of the updated node
  size_t current = (tree_index - 1) / 2;
  while (true) {
    size_t left_child = 2 * current + 1;
    size_t right_child = left_child + 1;

    float new_sum = sumTree_[left_child];
      // Check if right child is within tree bounds before accessing
    if (right_child < sumTree_.size()) {
      new_sum += sumTree_[right_child];
    }

    if (std::abs(sumTree_[current] - new_sum) < 1e-6 ) { // Tolerance check
      break; // Sum hasn't changed significantly
    }

    sumTree_[current] = new_sum;

    if (current == 0) {
      // Reached the root
      break;
    }
    current = (current - 1) / 2; // Move to parent
  }
}

void ReplayBuffer::updateTree(LeafIndexType leaf_index, float priority_alpha) {
  if (leaf_index >= capacity_) {
    throw std::out_of_range("Leaf index out of range in updateTree.");
  }
  // Index in the sumTree_ vector corresponding to the leaf
  size_t tree_index = leaf_index + capacity_ - 1;
  sumTree_[tree_index] = priority_alpha;
  propagate(tree_index);
}

std::pair<ReplayBuffer::LeafIndexType, float> ReplayBuffer::retrieve_leaf(float value_to_find) const {
  size_t current_node_idx = 0; // Start at the root (tree index)

  while (true) {
    size_t left_child_idx = 2 * current_node_idx + 1;
    size_t right_child_idx = left_child_idx + 1;

    if (left_child_idx >= sumTree_.size()) {
      // Reached a leaf
      break;
    }

    float left_child_value = sumTree_[left_child_idx];

    if (value_to_find <= left_child_value) {
      current_node_idx = left_child_idx; // Go left
    } else {
      value_to_find -= left_child_value; // Subtract left value
      if (right_child_idx < sumTree_.size()) {
        current_node_idx = right_child_idx; // Go right
      } else {
        // Should only happen if value > total priority or tree is corrupt
        current_node_idx = left_child_idx; // Fallback to last valid node
        break;
      }
    }
  }
  // Convert tree index back to leaf index (0 to capacity-1)
  LeafIndexType leaf_index = current_node_idx - (capacity_ - 1);
  float priority_alpha = sumTree_[current_node_idx];

  return {leaf_index, priority_alpha};
}

float ReplayBuffer::getTotalPriority() const {
  return sumTree_[0];
}

} // namespace rl