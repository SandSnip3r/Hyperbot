#ifndef PK2__REF__REGION_HPP_
#define PK2__REF__REGION_HPP_

#include <cstdint>
#include <string>

namespace sro::pk2::ref {

struct Region {
  int16_t wRegionID;
  uint8_t x;
  uint8_t z;
  std::string continentName;
  std::string areaName;
  uint8_t isBattleField;
  int32_t climate;
  int32_t maxCapacity;
  int32_t assocObjID;
  int32_t assocServer;
  std::string assocFile256;
  int32_t linkedRegion_1;
  int32_t linkedRegion_2;
  int32_t linkedRegion_3;
  int32_t linkedRegion_4;
  int32_t linkedRegion_5;
  int32_t linkedRegion_6;
  int32_t linkedRegion_7;
  int32_t linkedRegion_8;
  int32_t linkedRegion_9;
  int32_t linkedRegion_10;
};

} // namespace sro::pk2::ref

#endif // PK2__REF__REGION_HPP_