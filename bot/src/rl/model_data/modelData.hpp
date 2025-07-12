#ifndef RL_MODEL_DATA_MODELDATA_HPP_
#define RL_MODEL_DATA_MODELDATA_HPP_

#include <silkroad_lib/scalar_types.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace rl::model_data {

// ===============================================================================================
// ==================================== Structural Data Types ====================================
// ===============================================================================================

class BaseModelData {
protected:
  bool initialized_{false};
};

// Takes a current value, a min value, and a max value. Writes a normalized float in the range [0, 1].
template<typename T>
class NormalizedModelData : public BaseModelData {
public:
  static constexpr size_t size() { return 1; }

  size_t writeToArray(float* buffer) const;
  void setData(T currentValue, T minValue, T maxValue);
  T currentValue() const;
  T maxValue() const;
protected:
  T currentValue_;
  T minValue_;
  T maxValue_;
};

class BooleanModelData : public BaseModelData {
public:
  static constexpr size_t size() { return 1; }

  size_t writeToArray(float* buffer) const;
  void setData(bool value);
protected:
  bool data_;
};

template<size_t NumClasses>
class OneHotModelData : public BaseModelData {
public:
  static constexpr size_t size() { return NumClasses; }

  size_t writeToArray(float* buffer) const;
  void setData(size_t classIndex);
protected:
  size_t classIndex_;
};

// ===============================================================================================
// ===================================== Semantic Data Types =====================================
// ===============================================================================================

class SkillModelData {
public:
  // constexpr functions are implicitly inline, so we define them here rather than in the .cpp file.
  static constexpr size_t size() { return decltype(boolModelData_)::size(); }

  size_t writeToArray(float* buffer) const;
  // sro::scalar_types::ReferenceSkillId skillId() const { return skillId_; }
  void setSkillIsAvailable(sro::scalar_types::ReferenceSkillId skillId, bool isAvailable);
private:
  sro::scalar_types::ReferenceSkillId skillId_;
  BooleanModelData boolModelData_;
};

class ItemModelData {
public:
  // constexpr functions are implicitly inline, so we define them here rather than in the .cpp file.
  static constexpr size_t size() {
    return decltype(isAvailableModelData_)::size() +
           decltype(itemCountModelData_)::size();
  }

  size_t writeToArray(float* buffer) const;
  // sro::scalar_types::ReferenceObjectId itemId() const { return itemId_; }
  void setItemOnCooldownAndCount(sro::scalar_types::ReferenceObjectId itemId, bool isOnCooldown, uint16_t countAvailable, uint16_t maxCount);
private:
  sro::scalar_types::ReferenceObjectId itemId_;
  BooleanModelData isAvailableModelData_;
  NormalizedModelData<uint16_t> itemCountModelData_;
};

class VitalModelData {
public:
  static constexpr size_t size() { return decltype(normalizedModelData_)::size(); }

  size_t writeToArray(float* buffer) const;
  uint32_t currentValue() const { return normalizedModelData_.currentValue(); }
  uint32_t maxValue() const { return normalizedModelData_.maxValue(); }

  void setCurrentAndMax(uint32_t currentValue, uint32_t maxValue) {
    normalizedModelData_.setData(currentValue, 0, maxValue);
  }
private:
  NormalizedModelData<uint32_t> normalizedModelData_;
};

#include "modelData.inl"

} // namespace rl::model_data

#endif // RL_MODEL_DATA_MODELDATA_HPP_