#include "rl/replayBuffer.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

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

ReplayBuffer::ReplayBuffer(size_t capacity, size_t samplingBatchSize, float alpha, float beta, float epsilon)
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

ReplayBuffer::StorageIndexType ReplayBuffer::addObservationAndAction(common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId observerGlobalId, const Observation &observation, std::optional<int> actionIndex) {
  std::unique_lock lock{replayBufferMutex_};

  // TODO: Remove
  if (currentBufferSize_ == capacity_) {
    // Do not add if buffer is full, for now.
    return {};
  }

  // 1. Add transition to internal storage, get its index within that storage
  StorageIndexType storageIndex = storage_.addObservationAndAction(pvpId, observerGlobalId, observation, actionIndex);
  if (!storageIndex.havePrevious()) {
    // This is the first observation of this pvp for this player. We do not yet have a full transition. Nothing to do.
    return storageIndex;
  }

  // 2. Determine the leaf index in our sum-tree where this priority will go
  const LeafIndexType leafIndex = currentLeafIndex_;

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
    VLOG(1) << "Added transition #" << currentBufferSize_ << " to replay buffer.";
  }

  return storageIndex;
}

ObservationAndActionStorage::ObservationAndActionType ReplayBuffer::getObservationAndAction(StorageIndexType index) const {
  std::unique_lock lock{replayBufferMutex_};
  return storage_.getObservationAndAction(index);
}

std::vector<ReplayBuffer::SampleResult> ReplayBuffer::sample() {
  std::unique_lock lock{replayBufferMutex_};
  if (currentBufferSize_ < samplingBatchSize_) {
    throw std::runtime_error("Not enough transitions in buffer to sample a full batch.");
  }

  std::vector<SampleResult> results;
  results.reserve(samplingBatchSize_);

  const float totalPriority = getTotalPriority();
  if (totalPriority <= 0.0f) {
    throw std::runtime_error("Total priority is zero or negative, cannot sample.");
  }

  const float segmentSize = totalPriority / static_cast<float>(samplingBatchSize_);
  float maxWeight = 0.0f;

  for (size_t i = 0; i < samplingBatchSize_; ++i) {
    std::uniform_real_distribution<float> dist(i * segmentSize, (i + 1) * segmentSize);
    const float valueToFind = dist(rng_);

    // Retrieve leaf index (within sum-tree) and its stored priority (p^alpha)
    auto [leafIndex, priorityAlpha] = retrieveLeaf(valueToFind);

    // Get the corresponding storage index using the map
    const StorageIndexType observation1StorageIndex = leafToStorageMap_[leafIndex];

    // Calculate sampling probability P(i) = p_i^alpha / total_p_alpha
    float samplingProbability = priorityAlpha / totalPriority;
    if (samplingProbability <= 0.0f) {
      samplingProbability = std::numeric_limits<float>::epsilon(); // Avoid issues
    }

    // Calculate Importance Sampling (IS) weight: w_i = (N * P(i))^-beta
    // Note: Using currentBufferSize_ as N
    const float weight = std::pow(static_cast<float>(currentBufferSize_) * samplingProbability, -beta_);
    maxWeight = std::max(maxWeight, weight); // Track max weight for normalization

    // Create a new SampleResult object with one-lower index
    const StorageIndexType observation0StorageIndex = observation1StorageIndex.previous();

    results.push_back({observation1StorageIndex, TransitionType{storage_.getObservationAndAction(observation0StorageIndex), storage_.getObservationAndAction(observation1StorageIndex)}, weight});
  }

  // Normalize weights: w_i = w_i / max_w
  if (maxWeight > 0) {
    for (SampleResult &sample : results) {
      sample.weight /= maxWeight;
    }
  }

  return results;
}

void ReplayBuffer::updatePriorities(const std::vector<StorageIndexType>& storageIndices, const std::vector<float>& tdErrors) {
  std::unique_lock lock{replayBufferMutex_};
  if (storageIndices.size() != tdErrors.size()) {
    throw std::runtime_error("Indices and TD errors size mismatch in updatePriorities.");
  }

  for (size_t i=0; i<storageIndices.size(); ++i) {
    StorageIndexType storageIndex = storageIndices[i];
    auto it = storageToLeafMap_.find(storageIndex);

    if (it != storageToLeafMap_.end()) {
      LeafIndexType leafIndex = it->second; // Found the leaf index in the tree
      const float priorityAlpha = calculatePriority(tdErrors[i]);
      maxPriority_ = std::max(maxPriority_, priorityAlpha); // Ensure max_priority stays current
      updateTree(leafIndex, priorityAlpha);
    } else {
      // Handle cases where the storage index is not found in the map.
      // (e.g., it was overwritten and added again before update)
      throw std::runtime_error("Storage index not found in storageToLeafMap.");
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

float ReplayBuffer::calculatePriority(float tdError) const {
  return std::pow(std::abs(tdError) + epsilon_, alpha_);
}

void ReplayBuffer::propagate(size_t treeIndex) {
  // Start from parent of the updated node
  size_t current = (treeIndex - 1) / 2;
  while (true) {
    const size_t leftChild = 2 * current + 1;
    const size_t rightChild = leftChild + 1;

    float newSum = sumTree_[leftChild];
      // Check if right child is within tree bounds before accessing
    if (rightChild < sumTree_.size()) {
      newSum += sumTree_[rightChild];
    }
    sumTree_[current] = newSum;

    if (current == 0) {
      // Reached the root
      break;
    }
    current = (current - 1) / 2; // Move to parent
  }
}

void ReplayBuffer::updateTree(LeafIndexType leafIndex, float priorityAlpha) {
  if (leafIndex >= capacity_) {
    throw std::out_of_range("Leaf index out of range in updateTree.");
  }
  // Index in the sumTree_ vector corresponding to the leaf
  const size_t treeIndex = leafIndex + capacity_ - 1;
  sumTree_[treeIndex] = priorityAlpha;
  propagate(treeIndex);
}

std::pair<ReplayBuffer::LeafIndexType, float> ReplayBuffer::retrieveLeaf(float valueToFind) const {
  // Start at the root (tree index)
  size_t currentNodeIndex = 0;

  while (true) {
    const size_t leftChildIndex = 2 * currentNodeIndex + 1;
    const size_t rightChildIndex = leftChildIndex + 1;

    if (leftChildIndex >= sumTree_.size()) {
      // Reached a leaf
      break;
    }

    const float leftChildValue = sumTree_[leftChildIndex];

    if (valueToFind <= leftChildValue) {
      currentNodeIndex = leftChildIndex; // Go left
    } else {
      valueToFind -= leftChildValue; // Subtract left value
      if (rightChildIndex < sumTree_.size()) {
        currentNodeIndex = rightChildIndex; // Go right
      } else {
        // Should only happen if value > total priority or tree is corrupt
        throw std::runtime_error("Right child index out of bounds.");
      }
    }
  }

  // Convert tree index back to leaf index (0 to capacity-1)
  const LeafIndexType leafIndex = currentNodeIndex - (capacity_ - 1);
  const float priorityAlpha = sumTree_[currentNodeIndex];

  return {leafIndex, priorityAlpha};
}

float ReplayBuffer::getTotalPriority() const {
  return sumTree_[0];
}

} // namespace rl