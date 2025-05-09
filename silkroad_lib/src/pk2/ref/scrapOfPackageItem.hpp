#ifndef PK2_REF_SCRAP_OF_PACKAGE_ITEM_HPP_
#define PK2_REF_SCRAP_OF_PACKAGE_ITEM_HPP_

#include <array>
#include <cstdint>
#include <string>

namespace sro::pk2::ref {

struct ScrapOfPackageItem {
  uint8_t service;
  int32_t country;
  std::string refPackageItemCodeName;
  std::string refItemCodeName;
  uint8_t optLevel;
  int64_t variance;
  int32_t data;
  uint8_t magParamNum;
  std::array<int64_t, 12> magParams;
  int32_t param1;
  std::string param1Desc128;
  int32_t param2;
  std::string param2Desc128;
  int32_t param3;
  std::string param3Desc128;
  int32_t param4;
  std::string param4Desc128;
  int32_t index;
};

} // namespace sro::pk2::ref

#endif // PK2_REF_SCRAP_OF_PACKAGE_ITEM_HPP_