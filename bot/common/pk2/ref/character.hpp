#ifndef PK2_MEDIA_CHARACTER_HPP_
#define PK2_MEDIA_CHARACTER_HPP_

#include <cstdint>
#include <string>
#include <ostream>

namespace pk2::ref {

using CharacterId = int32_t;

struct Character {
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
	int16_t bCHeight;
	int16_t bCRadius;
	int32_t eventID;
	std::string assocFileObj128;
	std::string assocFileDrop128;
	std::string assocFileIcon128;
	std::string assocFile1_128;
	std::string assocFile2_128;
  uint8_t lvl;
  uint8_t charGender;
  int32_t maxHp;
  int32_t maxMp;
  // int32_t resistFrozen;
  // int32_t resistFrostbite;
  // int32_t resistBurn;
  // int32_t resistEShock;
  // int32_t resistPoison;
  // int32_t resistZombie;
  // int32_t resistSleep;
  // int32_t resistRoot;
  // int32_t resistSlow;
  // int32_t resistFear;
  // int32_t resistMyopia;
  // int32_t resistBlood;
  // int32_t resistStone;
  // int32_t resistDark;
  // int32_t resistStun;
  // int32_t resistDisea;
  // int32_t resistChaos;
  // int32_t resistCsePD;
  // int32_t resistCseMD;
  // int32_t resistCseSTR;
  // int32_t resistCseINT;
  // int32_t resistCseHP;
  // int32_t resistCseMP;
  // int32_t resist24;
  // int32_t resistBomb;
  // int32_t resist26;
  // int32_t resist27;
  // int32_t resist28;
  // int32_t resist29;
  // int32_t resist30;
  // int32_t resist31;
  // int32_t resist32;
  uint8_t inventorySize;
  uint8_t canStore_TID1;
  uint8_t canStore_TID2;
  uint8_t canStore_TID3;
  uint8_t canStore_TID4;
  uint8_t canBeVehicle;
  uint8_t canControl;
  uint8_t damagePortion;
  int16_t maxPassenger;
  int32_t assocTactics;
  int32_t pd;
  int32_t md;
  int32_t par;
  int32_t mar;
  int32_t er;
  int32_t br;
  int32_t hr;
  int32_t chr;
  int32_t expToGive;
  int32_t creepType;
  uint8_t knockdown;
  int32_t kO_RecoverTime;
  int32_t defaultSkill_1;
  int32_t defaultSkill_2;
  int32_t defaultSkill_3;
  int32_t defaultSkill_4;
  int32_t defaultSkill_5;
  int32_t defaultSkill_6;
  int32_t defaultSkill_7;
  int32_t defaultSkill_8;
  int32_t defaultSkill_9;
  int32_t defaultSkill_10;
  uint8_t textureType;
  int32_t except_1;
  int32_t except_2;
  int32_t except_3;
  int32_t except_4;
  int32_t except_5;
  int32_t except_6;
  int32_t except_7;
  int32_t except_8;
  int32_t except_9;
  int32_t except_10;
  // int32_t link;
};

std::ostream& operator<<(std::ostream &stream, const Character &character);

} // namespace pk2::ref

#endif // PK2_MEDIA_CHARACTER_HPP_