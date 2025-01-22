#ifndef PK2_MEDIA_TELEPORT_HPP_
#define PK2_MEDIA_TELEPORT_HPP_

#include <cstdint>
#include <string>

namespace sro::pk2::ref {

using TeleportId = int32_t;

struct Teleport {
  int32_t service;
  int32_t id;
  std::string codeName128;
  std::string objName128;
  std::string orgObjCodeName128;
  std::string nameStrID128;
  std::string descStrID128;
  uint8_t cashItem;
  uint8_t bionic;
  uint8_t typeId1;
  uint8_t typeId2;
  uint8_t typeId3;
  uint8_t typeId4;
  int32_t decayTime;
  uint8_t country;
  uint8_t rarity;
  uint8_t canTrade;
  uint8_t canSell;
  uint8_t canBuy;
  uint8_t canBorrow;
  uint8_t canDrop;
  uint8_t canPick;
  uint8_t canRepair;
  uint8_t canRevive;
  uint8_t canUse;
  uint8_t canThrow;
  int32_t price;
  int32_t costRepair;
  int32_t costRevive;
  int32_t costBorrow;
  int32_t keepingFee;
  int32_t sellPrice;
  int32_t reqLevelType1;
  uint8_t reqLevel1;
  int32_t reqLevelType2;
  uint8_t reqLevel2;
  int32_t reqLevelType3;
  uint8_t reqLevel3;
  int32_t reqLevelType4;
  uint8_t reqLevel4;
  int32_t maxContain;
  int16_t regionID;
  int16_t dir;
  int16_t offsetX;
  int16_t offsetY;
  int16_t offsetZ;
  int16_t speed1;
  int16_t speed2;
  int32_t scale;
  int16_t bcHeight;
  int16_t bcRadius;
  int32_t eventID;
  std::string assocFileObj128;
  std::string assocFileDrop128;
  std::string assocFileIcon128;
  std::string assocFile1_128;
  std::string assocFile2_128;
  int32_t link;
};

} // namespace sro::pk2::ref

#endif // PK2_MEDIA_TELEPORT_HPP_