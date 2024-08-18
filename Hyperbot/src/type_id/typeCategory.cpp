#include "typeCategory.hpp"

#include "helpers.hpp"

#include <absl/strings/str_format.h>

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

bool TypeCategory::isConcreteItem() const {
  return kTypeId4Mask == typeIdMask_;
}

TypeId TypeCategory::getTypeId() const {
  if (!isConcreteItem()) {
    throw std::runtime_error("Cannot get TypeId for non-concrete item");
  }
  return typeIdData_;
}

std::string toString(TypeId typeId) {
  const auto [tId1, tId2, tId3, tId4] = helpers::type_id::splitTypeId(typeId);
  return absl::StrFormat("%d,%d,%d,%d", tId1, tId2, tId3, tId4);
}

uint16_t makeTypeId(const uint16_t typeId1, const uint16_t typeId2, const uint16_t typeId3, const uint16_t typeId4) {
  return (typeId1 << 2) |
         (typeId2 << 5) |
         (typeId3 << 7) |
         (typeId4 << 11);
}

TypeId getTypeId(const pk2::ref::Character &character) {
  return makeTypeId(character.typeId1, character.typeId2, character.typeId3, character.typeId4);
}

TypeId getTypeId(const pk2::ref::Item &item) {
  return makeTypeId(item.typeId1, item.typeId2, item.typeId3, item.typeId4);
}

TypeId getTypeId(const pk2::ref::Teleport &teleport) {
  return makeTypeId(teleport.typeId1, teleport.typeId2, teleport.typeId3, teleport.typeId4);
}

} // namespace type_id