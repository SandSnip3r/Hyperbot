template<typename T>
size_t NormalizedModelData<T>::writeToArray(float* buffer) const {
  // if (!initialized_) {
  //   throw std::runtime_error("NormalizedModelData: Not initialized");
  // }
  buffer[0] = std::clamp((currentValue_ - minValue_) / static_cast<float>(maxValue_ - minValue_), 0.0f, 1.0f);
  return 1;
}

template<typename T>
void NormalizedModelData<T>::setData(T currentValue, T minValue, T maxValue) {
  if (minValue >= maxValue) {
    throw std::invalid_argument("NormalizedModelData: minValue must be less than maxValue");
  }
  currentValue_ = currentValue;
  minValue_ = minValue;
  maxValue_ = maxValue;
  initialized_ = true; // Mark the data as initialized
}

template<typename T>
T NormalizedModelData<T>::currentValue() const {
  if (!initialized_) {
    throw std::runtime_error("NormalizedModelData: Not initialized");
  }
  return currentValue_;
}

template<typename T>
T NormalizedModelData<T>::maxValue() const {
  if (!initialized_) {
    throw std::runtime_error("NormalizedModelData: Not initialized");
  }
  return maxValue_;
}

// ------------------------------------------------------------------------------------------------

template<size_t NumClasses>
size_t OneHotModelData<NumClasses>::writeToArray(float* buffer) const {
  // if (!initialized_) {
  //   throw std::runtime_error("OneHotModelData: Not initialized");
  // }
  std::fill(buffer, buffer + NumClasses, 0.0f);
  // TODO: On the setting of data, ensure classIndex_ is within the range [0, NumClasses - 1]
  if (classIndex_ < NumClasses) {
    buffer[classIndex_] = 1.0f;
  }
  return NumClasses;
}

template<size_t NumClasses>
void OneHotModelData<NumClasses>::setData(size_t classIndex) {
  if (classIndex >= NumClasses) {
    throw std::out_of_range("OneHotModelData: classIndex out of range");
  }
  classIndex_ = classIndex;
  initialized_ = true; // Mark the data as initialized
}