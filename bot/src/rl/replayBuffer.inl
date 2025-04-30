template<typename TransitionType>
ReplayBuffer<TransitionType>::ReplayBuffer(size_t capacity, float alpha, float beta, float epsilon)
      : capacity_(capacity),
        alpha_(alpha),
        beta_(beta),
        epsilon_(epsilon),
        sumTree_(2 * capacity - 1, 0.0f)
{
  if (alpha_ < 0.0f || alpha_ > 1.0f) {
    throw std::invalid_argument("Alpha must be in the range [0, 1].");
  }
  if (beta_ < 0.0f || beta_ > 1.0f) {
    throw std::invalid_argument("Beta must be in the range [0, 1].");
  }
  if (epsilon_ < 0.0f) {
    throw std::invalid_argument("Epsilon must be non-negative.");
  }
}

template<typename TransitionType>
typename ReplayBuffer<TransitionType>::TransitionId ReplayBuffer<TransitionType>::addTransition(const TransitionType &transition) {
  std::unique_lock lock{replayBufferMutex_};

  if (currentBufferSize_ == capacity_) {
    // Note that it is the user's responsibility to ensure that there is enough room before adding a transition.
    throw std::runtime_error("Replay buffer is full. Cannot add new transition.");
  }

  // Determine the leaf index in our sum-tree where this priority will go
  const LeafIndexType leafIndex = getNextLeafIndex();

  // Update the sum-tree using the max priority for the new transition. This is suggested by the paper so that new transitions are always sampled at least once.
  updateTreeForChangedLeaf(leafIndex, maxPriority_);

  currentBufferSize_++;
  advanceNextLeafIndex();

  return leafIndexToTransitionId(leafIndex);
}

template<typename TransitionType>
void ReplayBuffer<TransitionType>::deleteTransition(TransitionId id) {
  std::unique_lock lock{replayBufferMutex_};
  if (nextLeafIndex_) {
    // When deleting while nextLeafIndex_ is set, we need to change from using the next index to using the queue. To do that, we push all remaining sequential indices to the queue.
    // This could be very costly if we are far from full.
    for (LeafIndexType i = *nextLeafIndex_; i < capacity_; ++i) {
      nextLeafIndices_.push_back(i);
    }
    nextLeafIndex_.reset();
  }
  if (currentBufferSize_ == 0) {
    throw std::runtime_error("Replay buffer is empty. Cannot delete transition.");
  }
  const LeafIndexType leafIndex = transitionIdToLeafIndex(id);
  if (leafIndex >= capacity_) {
    throw std::out_of_range("Leaf index out of range in deleteTransition.");
  }
  updateTreeForChangedLeaf(leafIndex, 0.0f); // Set priority to 0
  currentBufferSize_--;
  nextLeafIndices_.push_back(leafIndex); // Add to free indices
}

template<typename TransitionType>
std::vector<typename ReplayBuffer<TransitionType>::SampleResult> ReplayBuffer<TransitionType>::sample(int count, std::mt19937 &rng) {
  std::unique_lock lock{replayBufferMutex_};
  if (count > currentBufferSize_) {
    throw std::runtime_error("Not enough transitions in buffer to sample a full batch.");
  }

  std::vector<SampleResult> results;
  results.reserve(count);

  const float totalPriority = getTotalPriority();
  if (totalPriority <= 0.0f) {
    throw std::runtime_error("Total priority is zero or negative, cannot sample.");
  }

  const float segmentSize = totalPriority / static_cast<float>(count);
  float maxWeight = 0.0f;

  for (size_t i = 0; i < count; ++i) {
    std::uniform_real_distribution<float> dist(i * segmentSize, (i + 1) * segmentSize);
    const float valueToFind = dist(rng);

    // Retrieve leaf index (within sum-tree) and its stored priority (p^alpha)
    const auto [leafIndex, priorityAlpha] = retrieveLeaf(valueToFind);

    // Calculate sampling probability P(i) = p_i^alpha / total_p_alpha
    float samplingProbability = priorityAlpha / totalPriority;
    if (samplingProbability <= 0.0f) {
      samplingProbability = std::numeric_limits<float>::epsilon(); // Avoid issues
    }

    // Calculate Importance Sampling (IS) weight: w_i = (N * P(i))^-beta
    // Note: Using currentBufferSize_ as N
    const float weight = std::pow(static_cast<float>(currentBufferSize_) * samplingProbability, -beta_);
    maxWeight = std::max(maxWeight, weight); // Track max weight for normalization

    results.push_back({leafIndexToTransitionId(leafIndex), weight});
  }

  // Normalize weights: w_i = w_i / max_w
  if (maxWeight > 0) {
    for (SampleResult &sample : results) {
      sample.weight /= maxWeight;
    }
  }

  return results;
}

template<typename TransitionType>
void ReplayBuffer<TransitionType>::updatePriorities(const std::vector<ReplayBuffer<TransitionType>::TransitionId> &ids, const std::vector<float> &priorities) {
  std::unique_lock lock{replayBufferMutex_};
  if (ids.size() != priorities.size()) {
    throw std::runtime_error("Size of ids does not match size of priorities in ReplayBuffer::updatePriorities.");
  }
  if (ids.size() > internalSize()) {
    throw std::runtime_error("Given too many priorities to update in ReplayBuffer::updatePriorities.");
  }

  for (size_t i=0; i<ids.size(); ++i) {
    LeafIndexType leafIndex = transitionIdToLeafIndex(ids[i]);
    const float priorityAlpha = calculatePriority(priorities[i]);
    maxPriority_ = std::max(maxPriority_, priorityAlpha); // Ensure max_priority stays current
    updateTreeForChangedLeaf(leafIndex, priorityAlpha);
  }
}

template<typename TransitionType>
size_t ReplayBuffer<TransitionType>::size() const {
  std::unique_lock lock{replayBufferMutex_};
  return internalSize();
}

// ========================= Private Methods =========================

template<typename TransitionType>
typename ReplayBuffer<TransitionType>::LeafIndexType ReplayBuffer<TransitionType>::getNextLeafIndex() const {
  if (nextLeafIndex_) {
    return *nextLeafIndex_;
  }
  if (nextLeafIndices_.empty()) {
    throw std::runtime_error("No free indices available in the queue.");
  }
  return nextLeafIndices_.front();
}

template<typename TransitionType>
void ReplayBuffer<TransitionType>::advanceNextLeafIndex() {
  if (nextLeafIndex_) {
    if (currentBufferSize_ == capacity_) {
      // Buffer is full, start using a queue for free indices
      nextLeafIndex_.reset();
    } else {
      ++(*nextLeafIndex_);
    }
  } else {
    if (nextLeafIndices_.empty()) {
      throw std::runtime_error("No free indices available in the queue.");
    }
    nextLeafIndices_.pop_front();
  }
}

template<typename TransitionType>
float ReplayBuffer<TransitionType>::calculatePriority(float priority) const {
  return std::pow(std::abs(priority) + epsilon_, alpha_);
}

template<typename TransitionType>
void ReplayBuffer<TransitionType>::propagateUpward(size_t treeIndex) {
  if (treeIndex == 0) {
    // Already at the root
    return;
  }
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

template<typename TransitionType>
void ReplayBuffer<TransitionType>::updateTreeForChangedLeaf(LeafIndexType leafIndex, float priorityAlpha) {
  if (leafIndex >= capacity_) {
    throw std::out_of_range("Leaf index out of range in updateTreeForChangedLeaf.");
  }
  // Index in the sumTree_ vector corresponding to the leaf
  const size_t treeIndex = leafIndex + capacity_ - 1;
  sumTree_[treeIndex] = priorityAlpha;
  propagateUpward(treeIndex);
}

template<typename TransitionType>
std::pair<typename ReplayBuffer<TransitionType>::LeafIndexType, float> ReplayBuffer<TransitionType>::retrieveLeaf(float valueToFind) const {
  // Start at the root (tree index)
  size_t currentNodeIndex = 0;

  // TODO: Remove once done debugging -->
  std::vector<size_t> indices;
  std::vector<float> values;
  indices.push_back(currentNodeIndex);
  values.push_back(valueToFind);
  // <--

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
      if (rightChildIndex < sumTree_.size()) {
        // Subtract left value and ensure it stays within range.
        valueToFind = std::min(valueToFind-leftChildValue, sumTree_[rightChildIndex]);
        // valueToFind = std::max(0.0f, std::min(valueToFind-leftChildValue, sumTree_[rightChildIndex]));
        currentNodeIndex = rightChildIndex; // Go right
      } else {
        // Should only happen if value > total priority or tree is corrupt
        throw std::runtime_error("Right child index out of bounds.");
      }
    }
    // TODO: Remove once done debugging -->
    indices.push_back(currentNodeIndex);
    values.push_back(valueToFind);
    // <--
  }

  // Convert tree index back to leaf index (0 to capacity-1)
  const LeafIndexType leafIndex = currentNodeIndex - (capacity_ - 1);
  if (leafIndex > internalSize()) {
    LOG(ERROR) << "Leaf index out of range: " << leafIndex << " > " << internalSize();
    LOG(ERROR) << "Indices & values:";
    for (size_t i = 0; i < indices.size(); ++i) {
      LOG(ERROR) << "  Index: " << indices[i] << ", Value: " << absl::StreamFormat("%17g", values[i]);
    }
    throw std::out_of_range(absl::StrFormat("Leaf index %d out of range (size: %d) in retrieveLeaf(%f).", leafIndex, internalSize(), valueToFind));
  }
  const float priorityAlpha = sumTree_[currentNodeIndex];

  return {leafIndex, priorityAlpha};
}

template<typename TransitionType>
float ReplayBuffer<TransitionType>::getTotalPriority() const {
  return sumTree_[0];
}