#include "typeCategory.hpp"

#include <stdexcept>

namespace type_id {

TypeCategory::TypeCategory(uint8_t typeId1) : typeIdData_((typeId1 & static_cast<TypeId>(0b111)) << 2) {}

TypeCategory::TypeCategory(TypeId typeIdData, TypeId typeIdMask) : typeIdData_(typeIdData), typeIdMask_(typeIdMask) {}

TypeCategory TypeCategory::subCategory(uint8_t nextTypeId) const {
  if (typeIdMask_ == kTypeId1Mask) {
    // Adding TypeId2 subcategory
    return TypeCategory{static_cast<TypeId>(typeIdData_ | ((nextTypeId & static_cast<TypeId>(0b11)) << 5)), kTypeId2Mask};
  } else if (typeIdMask_ == kTypeId2Mask) {
    // Adding TypeId3 subcategory
    return TypeCategory{static_cast<TypeId>(typeIdData_ | ((nextTypeId & static_cast<TypeId>(0b1111)) << 7)), kTypeId3Mask};
  } else /* if (typeIdMask_ == kTypeId3Mask) */ {
    // Adding TypeId4 subcategory
    return TypeCategory{static_cast<TypeId>(typeIdData_ | ((nextTypeId & static_cast<TypeId>(0b11111)) << 11)), kTypeId4Mask};
  }
  throw std::runtime_error("Cannot add another subcategory");
}

bool TypeCategory::contains(TypeId typeId) const {
  return (typeId & typeIdMask_) == (typeIdData_ & typeIdMask_);
}

bool TypeCategory::contains(const TypeCategory typeCategory) const {
  // A larger mask value means a more specific category
  if (this->typeIdMask_ > typeCategory.typeIdMask_) {
    // Our category is more specific than the given category
    return false;
  }
  return contains(typeCategory.typeIdData_);
}

} // namespace type_id