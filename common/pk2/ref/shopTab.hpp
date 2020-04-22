#ifndef PK2_REF_SHOP_TAB_HPP_
#define PK2_REF_SHOP_TAB_HPP_

#include <string>

namespace pk2::ref {

struct ShopTab {
  uint8_t service;
  int32_t country;
	int32_t id;
  std::string codeName128;
  std::string refTabGroupCodeName;
  std::string strID128Tab;
};

} // namespace pk2::ref

#endif // PK2_REF_SHOP_TAB_HPP_