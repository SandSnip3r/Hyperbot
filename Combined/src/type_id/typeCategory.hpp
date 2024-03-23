#ifndef TYPE_ID_TYPE_CATEGORY_HPP_
#define TYPE_ID_TYPE_CATEGORY_HPP_

#include "../../common/pk2/ref/character.hpp"
#include "../../common/pk2/ref/item.hpp"
#include "../../common/pk2/ref/teleport.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace type_id {

// (MSB)                                                                       (LSB)
// | 15 | 14 | 13 | 12 | 11 | 10 | 09 | 08 | 07 | 06 | 05 | 04 | 03 | 02 | 01 | 00 |
// |         TypeID4        |       TypeID3     | TypeID2 |    TypeID1   | BI | CI |
// BI - Bionic
// CI - CashItem
using TypeId = uint16_t;

class TypeCategory {
private:
  static constexpr TypeId kTypeId1Mask{0b11100};
  static constexpr TypeId kTypeId2Mask{0b1111100};
  static constexpr TypeId kTypeId3Mask{0b11111111100};
  static constexpr TypeId kTypeId4Mask{0b1111111111111100};
public:
  explicit TypeCategory(uint8_t typeId1);
  explicit TypeCategory(TypeId typeIdData, TypeId typeIdMask = kTypeId4Mask);
  TypeCategory subCategory(uint8_t nextTypeId) const;
  bool contains(TypeId typeId) const;
  bool contains(const TypeCategory typeCategory) const;

  template<typename T>
  bool contains(T t) const = delete;

  bool isConcreteItem() const;
  TypeId getTypeId() const;
private:
  const TypeId typeIdData_{0};
  const TypeId typeIdMask_{kTypeId1Mask};
};

std::string toString(TypeId typeId);

uint16_t makeTypeId(const uint16_t typeId1, const uint16_t typeId2, const uint16_t typeId3, const uint16_t typeId4);

TypeId getTypeId(const pk2::ref::Character &character);
TypeId getTypeId(const pk2::ref::Item &item);
TypeId getTypeId(const pk2::ref::Teleport &teleport);

} // namespace type_id

#endif // TYPE_ID_TYPE_CATEGORY_HPP_