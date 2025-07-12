#include "modelData.hpp"

namespace rl::model_data {

// ===============================================================================================
// ==================================== Structural Data Types ====================================
// ===============================================================================================

size_t BooleanModelData::writeToArray(float* buffer) const {
  // if (!initialized_) {
  //   throw std::runtime_error("BooleanModelData: Not initialized");
  // }
  buffer[0] = data_ ? 1.0f : 0.0f;
  return 1;
}

void BooleanModelData::setData(bool value) {
  data_ = value;
  initialized_ = true; // Mark the data as initialized
}

// ===============================================================================================
// ===================================== Semantic Data Types =====================================
// ===============================================================================================

size_t SkillModelData::writeToArray(float* buffer) const {
  return boolModelData_.writeToArray(buffer);
}

void SkillModelData::setSkillIsAvailable(sro::scalar_types::ReferenceSkillId skillId, bool isAvailable) {
  skillId_ = skillId;
  boolModelData_.setData(isAvailable);
}

// ------------------------------------------------------------------------------------------------

size_t ItemModelData::writeToArray(float* buffer) const {
  size_t bytesWritten = 0;
  bytesWritten += isAvailableModelData_.writeToArray(&buffer[bytesWritten]);
  bytesWritten += itemCountModelData_.writeToArray(&buffer[bytesWritten]);
  return bytesWritten;
}

void ItemModelData::setItemOnCooldownAndCount(sro::scalar_types::ReferenceObjectId itemId, bool isOnCooldown, uint16_t countAvailable, uint16_t maxCount) {
  itemId_ = itemId;
  isAvailableModelData_.setData(!isOnCooldown && countAvailable > 0);
  itemCountModelData_.setData(countAvailable, /*minValue=*/0, maxCount);
}

// ------------------------------------------------------------------------------------------------

size_t VitalModelData::writeToArray(float* buffer) const {
  return normalizedModelData_.writeToArray(buffer);
}

} // namespace rl::model_data