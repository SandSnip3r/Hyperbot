#ifndef PK2_REF_SHOP_GROUP_HPP_
#define PK2_REF_SHOP_GROUP_HPP_

#include <string>

namespace sro::pk2::ref {

// using ShopGroupId = int16_t;

struct ShopGroup {
  uint8_t service;
  int32_t country;
  int16_t id;
  std::string codeName128;
  std::string refNPCCodeName;
  int32_t param1;
  std::string param1Desc128;
  int32_t param2;
  std::string param2Desc128;
  int32_t param3;
  std::string param3Desc128;
  int32_t param4;
  std::string param4Desc128;
};

} // namespace sro::pk2::ref

#endif // PK2_REF_SHOP_GROUP_HPP_