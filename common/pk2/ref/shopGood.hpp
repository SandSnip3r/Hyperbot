#ifndef PK2_REF_SHOP_GOOD_HPP_
#define PK2_REF_SHOP_GOOD_HPP_

#include <string>

namespace pk2::ref {

struct ShopGood {
  // uint8_t service
  // int32_t country
  std::string refTabCodeName;
  std::string refPackageItemCodeName;
  uint8_t slotIndex;
  // int32_t param1
  // std::string param1_Desc128
  // int32_t param2
  // std::string param2_Desc128
  // int32_t param3
  // std::string param3_Desc128
  // int32_t param4
  // std::string param4_Desc128
};

} // namespace pk2::ref

#endif // PK2_REF_SHOP_GOOD_HPP_