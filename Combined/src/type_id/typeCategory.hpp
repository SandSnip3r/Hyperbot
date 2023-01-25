#ifndef TYPE_ID_TYPE_CATEGORY_HPP_
#define TYPE_ID_TYPE_CATEGORY_HPP_

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
public:
  TypeCategory(uint8_t typeId1);
  TypeCategory subCategory(uint8_t nextTypeId) const;
  bool contains(TypeId typeId) const;
  bool contains(const TypeCategory typeCategory) const;
  bool isConcreteItem() const;
  TypeId getTypeId() const;
private:
  static constexpr TypeId kTypeId1Mask{0b11100};
  static constexpr TypeId kTypeId2Mask{0b1111100};
  static constexpr TypeId kTypeId3Mask{0b11111111100};
  static constexpr TypeId kTypeId4Mask{0b1111111111111100};
  const TypeId typeIdData_{0};
  const TypeId typeIdMask_{kTypeId1Mask};
  TypeCategory(TypeId typeIdData, TypeId typeIdMask);
};

std::string toString(TypeId typeId);

} // namespace type_id

#endif // TYPE_ID_TYPE_CATEGORY_HPP_