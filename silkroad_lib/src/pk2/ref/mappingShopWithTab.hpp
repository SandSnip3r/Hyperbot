#ifndef PK2_REF_MAP_SHOP_WITH_TAB_HPP_
#define PK2_REF_MAP_SHOP_WITH_TAB_HPP_

#include <cstdint>
#include <string>

namespace sro::pk2::ref {

struct MappingShopWithTab {
  uint8_t service;
  int32_t country;
  std::string refShopCodeName;
  std::string refTabGroupCodeName;
};

} // namespace sro::pk2::ref

#endif // PK2_REF_MAP_SHOP_WITH_TAB_HPP_