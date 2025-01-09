#ifndef PK2_REF_MAPPING_SHOP_GROUP_HPP_
#define PK2_REF_MAPPING_SHOP_GROUP_HPP_

#include <string>

namespace pk2::ref {

struct MappingShopGroup {
  uint8_t service;
  int32_t country;
  std::string refShopGroupCodeName;
  std::string refShopCodeName;
};

} // namespace pk2::ref

#endif // PK2_REF_MAPPING_SHOP_GROUP_HPP_