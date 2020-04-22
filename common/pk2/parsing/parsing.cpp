#include "parsing.hpp"

namespace pk2::parsing {

std::string fileDataToString(const std::vector<uint8_t> &data) {
  // TODO: This is UTF16-LE, Improve this function using a proper conversion
  //  This function takes roughly twice as long as the parsing function
  //  PK2Reader::getEntryData took 1ms
  //  This function took 526ms
  //  Parsing took 299ms    
  // const auto size = data.size()-2;
  // std::u16string u16((size/2)+1, '\0');
  // std::memcpy(u16.data(), data.data()+2, size);
  // return std::wstring_convert<
  //     std::codecvt_utf8_utf16<char16_t>, char16_t>{}.to_bytes(u16);
  // // https://stackoverflow.com/a/56723923/1148866

	if (data.size()%2 != 0) {
		throw std::runtime_error("Data is not evenly sized");
	}
	std::string result;
	result.reserve((data.size()-2)/2);
  // Skipping first 2 bytes for BOM
	for (int i=2; i<data.size(); i+=2) {
		result += (char)data[i];
	}
	return result;
}

namespace {
bool isValidLine(const int fieldCount, const std::string &line) {
  // TODO: Evaluate more robust way to validate
  if (line.size() < fieldCount*2+1) {
    return false;
  }
  if (line[0] == '/' && line[1] == '/') {
    return false;
  }
  return true;
}
}

bool isValidCharacterdataLine(const std::string &line) {
  constexpr int kDataCount = 137;
  return isValidLine(kDataCount, line);
}

bool isValidItemdataLine(const std::string &line) {
  constexpr int kDataCount = 160;
  return isValidLine(kDataCount, line);
}

bool isValidSkilldataLine(const std::string &line) {
  constexpr int kDataCount = 117;
  return isValidLine(kDataCount, line);
}

bool isValidTeleportbuildingLine(const std::string &line) {
  constexpr int kDataCount = 58;
  return isValidLine(kDataCount, line);
}

bool isValidScrapOfPackageItemLine(const std::string &line) {
  constexpr int kDataCount = 29;
  return isValidLine(kDataCount, line);
}

bool isValidShopTabLine(const std::string &line) {
  constexpr int kDataCount = 6;
  return isValidLine(kDataCount, line);
}

bool isValidShopGroupLine(const std::string &line) {
  constexpr int kDataCount = 13;
  return isValidLine(kDataCount, line);
}

bool isValidShopGoodLine(const std::string &line) {
  constexpr int kDataCount = 13;
  return isValidLine(kDataCount, line);
}
bool isValidMappingShopGroupLine(const std::string &line) {
  constexpr int kDataCount = 4;
  return isValidLine(kDataCount, line);
}

bool isValidMappingShopWithTabLine(const std::string &line) {
  constexpr int kDataCount = 4;
  return isValidLine(kDataCount, line);
}

pk2::ref::Character parseCharacterdataLine(const std::string &line) {
	pk2::ref::Character character;
  const char *ptr = line.data();
	ptr = parse(ptr, character.service);
	ptr = parse(ptr, character.id);
	ptr = parse(ptr, character.codeName128);
	ptr = parse(ptr, character.objName128);
	ptr = parse(ptr, character.orgObjCodeName128);
	ptr = parse(ptr, character.nameStrID128);
	ptr = parse(ptr, character.descStrID128);
	ptr = parse(ptr, character.cashItem);
	ptr = parse(ptr, character.bionic);
	ptr = parse(ptr, character.typeId1);
	ptr = parse(ptr, character.typeId2);
	ptr = parse(ptr, character.typeId3);
	ptr = parse(ptr, character.typeId4);
	ptr = parse(ptr, character.decayTime);
	ptr = parse(ptr, character.country);
	ptr = parse(ptr, character.rarity);
	ptr = parse(ptr, character.canTrade);
	ptr = parse(ptr, character.canSell);
	ptr = parse(ptr, character.canBuy);
	ptr = parse(ptr, character.canBorrow);
	ptr = parse(ptr, character.canDrop);
	ptr = parse(ptr, character.canPick);
	ptr = parse(ptr, character.canRepair);
	ptr = parse(ptr, character.canRevive);
	ptr = parse(ptr, character.canUse);
	ptr = parse(ptr, character.canThrow);
	ptr = parse(ptr, character.price);
	ptr = parse(ptr, character.costRepair);
	ptr = parse(ptr, character.costRevive);
	ptr = parse(ptr, character.costBorrow);
	ptr = parse(ptr, character.keepingFee);
	ptr = parse(ptr, character.sellPrice);
	ptr = parse(ptr, character.reqLevelType1);
	ptr = parse(ptr, character.reqLevel1);
	ptr = parse(ptr, character.reqLevelType2);
	ptr = parse(ptr, character.reqLevel2);
	ptr = parse(ptr, character.reqLevelType3);
	ptr = parse(ptr, character.reqLevel3);
	ptr = parse(ptr, character.reqLevelType4);
	ptr = parse(ptr, character.reqLevel4);
	ptr = parse(ptr, character.maxContain);
	ptr = parse(ptr, character.regionID);
	ptr = parse(ptr, character.dir);
	ptr = parse(ptr, character.offsetX);
	ptr = parse(ptr, character.offsetY);
	ptr = parse(ptr, character.offsetZ);
	ptr = parse(ptr, character.speed1);
	ptr = parse(ptr, character.speed2);
	ptr = parse(ptr, character.scale);
	ptr = parse(ptr, character.bCHeight);
	ptr = parse(ptr, character.bCRadius);
	ptr = parse(ptr, character.eventID);
	ptr = parse(ptr, character.assocFileObj128);
	ptr = parse(ptr, character.assocFileDrop128);
	ptr = parse(ptr, character.assocFileIcon128);
	ptr = parse(ptr, character.assocFile1_128);
	ptr = parse(ptr, character.assocFile2_128);
  ptr = parse(ptr, character.lvl);
  ptr = parse(ptr, character.charGender);
  ptr = parse(ptr, character.maxHP);
  ptr = parse(ptr, character.maxMP);
  ptr = parse(ptr, character.resistFrozen);
  ptr = parse(ptr, character.resistFrostbite);
  ptr = parse(ptr, character.resistBurn);
  ptr = parse(ptr, character.resistEShock);
  ptr = parse(ptr, character.resistPoison);
  ptr = parse(ptr, character.resistZombie);
  ptr = parse(ptr, character.resistSleep);
  ptr = parse(ptr, character.resistRoot);
  ptr = parse(ptr, character.resistSlow);
  ptr = parse(ptr, character.resistFear);
  ptr = parse(ptr, character.resistMyopia);
  ptr = parse(ptr, character.resistBlood);
  ptr = parse(ptr, character.resistStone);
  ptr = parse(ptr, character.resistDark);
  ptr = parse(ptr, character.resistStun);
  ptr = parse(ptr, character.resistDisea);
  ptr = parse(ptr, character.resistChaos);
  ptr = parse(ptr, character.resistCsePD);
  ptr = parse(ptr, character.resistCseMD);
  ptr = parse(ptr, character.resistCseSTR);
  ptr = parse(ptr, character.resistCseINT);
  ptr = parse(ptr, character.resistCseHP);
  ptr = parse(ptr, character.resistCseMP);
  ptr = parse(ptr, character.resist24);
  ptr = parse(ptr, character.resistBomb);
  ptr = parse(ptr, character.resist26);
  ptr = parse(ptr, character.resist27);
  ptr = parse(ptr, character.resist28);
  ptr = parse(ptr, character.resist29);
  ptr = parse(ptr, character.resist30);
  ptr = parse(ptr, character.resist31);
  ptr = parse(ptr, character.resist32);
  ptr = parse(ptr, character.inventorySize);
  ptr = parse(ptr, character.canStore_TID1);
  ptr = parse(ptr, character.canStore_TID2);
  ptr = parse(ptr, character.canStore_TID3);
  ptr = parse(ptr, character.canStore_TID4);
  ptr = parse(ptr, character.canBeVehicle);
  ptr = parse(ptr, character.canControl);
  ptr = parse(ptr, character.damagePortion);
  ptr = parse(ptr, character.maxPassenger);
  ptr = parse(ptr, character.assocTactics);
  ptr = parse(ptr, character.pd);
  ptr = parse(ptr, character.md);
  ptr = parse(ptr, character.par);
  ptr = parse(ptr, character.mar);
  ptr = parse(ptr, character.er);
  ptr = parse(ptr, character.br);
  ptr = parse(ptr, character.hr);
  ptr = parse(ptr, character.chr);
  ptr = parse(ptr, character.expToGive);
  ptr = parse(ptr, character.creepType);
  ptr = parse(ptr, character.knockdown);
  ptr = parse(ptr, character.kO_RecoverTime);
  ptr = parse(ptr, character.defaultSkill_1);
  ptr = parse(ptr, character.defaultSkill_2);
  ptr = parse(ptr, character.defaultSkill_3);
  ptr = parse(ptr, character.defaultSkill_4);
  ptr = parse(ptr, character.defaultSkill_5);
  ptr = parse(ptr, character.defaultSkill_6);
  ptr = parse(ptr, character.defaultSkill_7);
  ptr = parse(ptr, character.defaultSkill_8);
  ptr = parse(ptr, character.defaultSkill_9);
  ptr = parse(ptr, character.defaultSkill_10);
  ptr = parse(ptr, character.textureType);
  ptr = parse(ptr, character.except_1);
  ptr = parse(ptr, character.except_2);
  ptr = parse(ptr, character.except_3);
  ptr = parse(ptr, character.except_4);
  ptr = parse(ptr, character.except_5);
  ptr = parse(ptr, character.except_6);
  ptr = parse(ptr, character.except_7);
  ptr = parse(ptr, character.except_8);
  ptr = parse(ptr, character.except_9);
  ptr = parse(ptr, character.except_10);
  parse(ptr, character.link);
	return character;
}

pk2::ref::Item parseItemdataLine(const std::string &line) {
	pk2::ref::Item item;
  const char *ptr = line.data();
	ptr = parse(ptr, item.service);
	ptr = parse(ptr, item.id);
	ptr = parse(ptr, item.codeName128);
	ptr = parse(ptr, item.objName128);
	ptr = parse(ptr, item.orgObjCodeName128);
	ptr = parse(ptr, item.nameStrID128);
	ptr = parse(ptr, item.descStrID128);
	ptr = parse(ptr, item.cashItem);
	ptr = parse(ptr, item.bionic);
	ptr = parse(ptr, item.typeId1);
	ptr = parse(ptr, item.typeId2);
	ptr = parse(ptr, item.typeId3);
	ptr = parse(ptr, item.typeId4);
	ptr = parse(ptr, item.decayTime);
	ptr = parse(ptr, item.country);
	ptr = parse(ptr, item.rarity);
	ptr = parse(ptr, item.canTrade);
	ptr = parse(ptr, item.canSell);
	ptr = parse(ptr, item.canBuy);
	ptr = parse(ptr, item.canBorrow);
	ptr = parse(ptr, item.canDrop);
	ptr = parse(ptr, item.canPick);
	ptr = parse(ptr, item.canRepair);
	ptr = parse(ptr, item.canRevive);
	ptr = parse(ptr, item.canUse);
	ptr = parse(ptr, item.canThrow);
	ptr = parse(ptr, item.price);
	ptr = parse(ptr, item.costRepair);
	ptr = parse(ptr, item.costRevive);
	ptr = parse(ptr, item.costBorrow);
	ptr = parse(ptr, item.keepingFee);
	ptr = parse(ptr, item.sellPrice);
	ptr = parse(ptr, item.reqLevelType1);
	ptr = parse(ptr, item.reqLevel1);
	ptr = parse(ptr, item.reqLevelType2);
	ptr = parse(ptr, item.reqLevel2);
	ptr = parse(ptr, item.reqLevelType3);
	ptr = parse(ptr, item.reqLevel3);
	ptr = parse(ptr, item.reqLevelType4);
	ptr = parse(ptr, item.reqLevel4);
	ptr = parse(ptr, item.maxContain);
	ptr = parse(ptr, item.regionID);
	ptr = parse(ptr, item.dir);
	ptr = parse(ptr, item.offsetX);
	ptr = parse(ptr, item.offsetY);
	ptr = parse(ptr, item.offsetZ);
	ptr = parse(ptr, item.speed1);
	ptr = parse(ptr, item.speed2);
	ptr = parse(ptr, item.scale);
	ptr = parse(ptr, item.bCHeight);
	ptr = parse(ptr, item.bCRadius);
	ptr = parse(ptr, item.eventID);
	ptr = parse(ptr, item.assocFileObj128);
	ptr = parse(ptr, item.assocFileDrop128);
	ptr = parse(ptr, item.assocFileIcon128);
	ptr = parse(ptr, item.assocFile1_128);
	ptr = parse(ptr, item.assocFile2_128);
	ptr = parse(ptr, item.maxStack);
	ptr = parse(ptr, item.reqGender);
	ptr = parse(ptr, item.reqStr);
	ptr = parse(ptr, item.reqInt);
	ptr = parse(ptr, item.itemClass);
	ptr = parse(ptr, item.setID);
	ptr = parse(ptr, item.dur_L);
	ptr = parse(ptr, item.dur_U);
	ptr = parse(ptr, item.pd_L);
	ptr = parse(ptr, item.pd_U);
	ptr = parse(ptr, item.pdInc);
	ptr = parse(ptr, item.er_L);
	ptr = parse(ptr, item.er_U);
	ptr = parse(ptr, item.eRInc);
	ptr = parse(ptr, item.par_L);
	ptr = parse(ptr, item.par_U);
	ptr = parse(ptr, item.parInc);
	ptr = parse(ptr, item.br_L);
	ptr = parse(ptr, item.br_U);
	ptr = parse(ptr, item.md_L);
	ptr = parse(ptr, item.md_U);
	ptr = parse(ptr, item.mdInc);
	ptr = parse(ptr, item.mar_L);
	ptr = parse(ptr, item.mar_U);
	ptr = parse(ptr, item.marInc);
	ptr = parse(ptr, item.pdStr_L);
	ptr = parse(ptr, item.pdStr_U);
	ptr = parse(ptr, item.mdInt_L);
	ptr = parse(ptr, item.mdInt_U);
	ptr = parse(ptr, item.quivered);
	ptr = parse(ptr, item.ammo1_TID4);
	ptr = parse(ptr, item.ammo2_TID4);
	ptr = parse(ptr, item.ammo3_TID4);
	ptr = parse(ptr, item.ammo4_TID4);
	ptr = parse(ptr, item.ammo5_TID4);
	ptr = parse(ptr, item.speedClass);
	ptr = parse(ptr, item.twoHanded);
	ptr = parse(ptr, item.range);
	ptr = parse(ptr, item.pAttackMin_L);
	ptr = parse(ptr, item.pAttackMin_U);
	ptr = parse(ptr, item.pAttackMax_L);
	ptr = parse(ptr, item.pAttackMax_U);
	ptr = parse(ptr, item.pAttackInc);
	ptr = parse(ptr, item.mAttackMin_L);
	ptr = parse(ptr, item.mAttackMin_U);
	ptr = parse(ptr, item.mAttackMax_L);
	ptr = parse(ptr, item.mAttackMax_U);
	ptr = parse(ptr, item.mAttackInc);
	ptr = parse(ptr, item.paStrMin_L);
	ptr = parse(ptr, item.paStrMin_U);
	ptr = parse(ptr, item.paStrMax_L);
	ptr = parse(ptr, item.paStrMax_U);
	ptr = parse(ptr, item.maInt_Min_L);
	ptr = parse(ptr, item.maInt_Min_U);
	ptr = parse(ptr, item.maInt_Max_L);
	ptr = parse(ptr, item.maInt_Max_U);
	ptr = parse(ptr, item.hr_L);
	ptr = parse(ptr, item.hr_U);
	ptr = parse(ptr, item.hRInc);
	ptr = parse(ptr, item.cHR_L);
	ptr = parse(ptr, item.cHR_U);
	ptr = parse(ptr, item.param1);
	ptr = parse(ptr, item.desc1_128);
	ptr = parse(ptr, item.param2);
	ptr = parse(ptr, item.desc2_128);
	ptr = parse(ptr, item.param3);
	ptr = parse(ptr, item.desc3_128);
	ptr = parse(ptr, item.param4);
	ptr = parse(ptr, item.desc4_128);
	ptr = parse(ptr, item.param5);
	ptr = parse(ptr, item.desc5_128);
	ptr = parse(ptr, item.param6);
	ptr = parse(ptr, item.desc6_128);
	ptr = parse(ptr, item.param7);
	ptr = parse(ptr, item.desc7_128);
	ptr = parse(ptr, item.param8);
	ptr = parse(ptr, item.desc8_128);
	ptr = parse(ptr, item.param9);
	ptr = parse(ptr, item.desc9_128);
	ptr = parse(ptr, item.param10);
	ptr = parse(ptr, item.desc10_128);
	ptr = parse(ptr, item.param11);
	ptr = parse(ptr, item.desc11_128);
	ptr = parse(ptr, item.param12);
	ptr = parse(ptr, item.desc12_128);
	ptr = parse(ptr, item.param13);
	ptr = parse(ptr, item.desc13_128);
	ptr = parse(ptr, item.param14);
	ptr = parse(ptr, item.desc14_128);
	ptr = parse(ptr, item.param15);
	ptr = parse(ptr, item.desc15_128);
	ptr = parse(ptr, item.param16);
	ptr = parse(ptr, item.desc16_128);
	ptr = parse(ptr, item.param17);
	ptr = parse(ptr, item.desc17_128);
	ptr = parse(ptr, item.param18);
	ptr = parse(ptr, item.desc18_128);
	ptr = parse(ptr, item.param19);
	ptr = parse(ptr, item.desc19_128);
	ptr = parse(ptr, item.param20);
	ptr = parse(ptr, item.desc20_128);
	ptr = parse(ptr, item.maxMagicOptCount);
	parse(ptr, item.childItemCount);
	return item;
}

pk2::ref::Skill parseSkilldataLine(const std::string &line) {
	pk2::ref::Skill skill;
  const char *ptr = line.data();
  ptr = parse(ptr, skill.service);
  ptr = parse(ptr, skill.id);
  ptr = parse(ptr, skill.groupId);
  ptr = parse(ptr, skill.basicCode);
  ptr = parse(ptr, skill.basicName);
  ptr = parse(ptr, skill.basicGroup);
  ptr = parse(ptr, skill.basicOriginal);
  ptr = parse(ptr, skill.basicLevel);
  ptr = parse(ptr, skill.basicActivity);
  ptr = parse(ptr, skill.basicChainCode);
  ptr = parse(ptr, skill.basicRecycleCost);
  ptr = parse(ptr, skill.actionPreparingTime);
  ptr = parse(ptr, skill.actionCastingTime);
  ptr = parse(ptr, skill.actionActionDuration);
  ptr = parse(ptr, skill.actionReuseDelay);
  ptr = parse(ptr, skill.actionCoolTime);
  ptr = parse(ptr, skill.actionFlyingSpeed);
  ptr = parse(ptr, skill.actionInterruptable);
  ptr = parse(ptr, skill.actionOverlap);
  ptr = parse(ptr, skill.actionAutoAttackType);
  ptr = parse(ptr, skill.actionInTown);
  ptr = parse(ptr, skill.actionRange);
  ptr = parse(ptr, skill.targetRequired);
  ptr = parse(ptr, skill.targetTypeAnimal);
  ptr = parse(ptr, skill.targetTypeLand);
  ptr = parse(ptr, skill.targetTypeBuilding);
  ptr = parse(ptr, skill.targetGroupSelf);
  ptr = parse(ptr, skill.targetGroupAlly);
  ptr = parse(ptr, skill.targetGroupParty);
  ptr = parse(ptr, skill.targetGroupEnemy_M);
  ptr = parse(ptr, skill.targetGroupEnemy_P);
  ptr = parse(ptr, skill.targetGroupNeutral);
  ptr = parse(ptr, skill.targetGroupDontCare);
  ptr = parse(ptr, skill.targetEtcSelectDeadBody);
  ptr = parse(ptr, skill.reqCommonMastery1);
  ptr = parse(ptr, skill.reqCommonMastery2);
  ptr = parse(ptr, skill.reqCommonMasteryLevel1);
  ptr = parse(ptr, skill.reqCommonMasteryLevel2);
  ptr = parse(ptr, skill.reqCommonStr);
  ptr = parse(ptr, skill.reqCommonInt);
  ptr = parse(ptr, skill.reqLearnSkill1);
  ptr = parse(ptr, skill.reqLearnSkill2);
  ptr = parse(ptr, skill.reqLearnSkill3);
  ptr = parse(ptr, skill.reqLearnSkillLevel1);
  ptr = parse(ptr, skill.reqLearnSkillLevel2);
  ptr = parse(ptr, skill.reqLearnSkillLevel3);
  ptr = parse(ptr, skill.reqLearnSP);
  ptr = parse(ptr, skill.reqLearnRace);
  ptr = parse(ptr, skill.reqRestriction1);
  ptr = parse(ptr, skill.reqRestriction2);
  ptr = parse(ptr, skill.reqCastWeapon1);
  ptr = parse(ptr, skill.reqCastWeapon2);
  ptr = parse(ptr, skill.consumeHP);
  ptr = parse(ptr, skill.consumeMP);
  ptr = parse(ptr, skill.consumeHPRatio);
  ptr = parse(ptr, skill.consumeMPRatio);
  ptr = parse(ptr, skill.consumeWHAN);
  ptr = parse(ptr, skill.uiSkillTab);
  ptr = parse(ptr, skill.uiSkillPage);
  ptr = parse(ptr, skill.uiSkillColumn);
  ptr = parse(ptr, skill.uiSkillRow);
  ptr = parse(ptr, skill.uiIconFile);
  ptr = parse(ptr, skill.uiSkillName);
  ptr = parse(ptr, skill.uiSkillToolTip);
  ptr = parse(ptr, skill.uiSkillToolTip_Desc);
  ptr = parse(ptr, skill.uiSkillStudy_Desc);
  ptr = parse(ptr, skill.aiAttackChance);
  ptr = parse(ptr, skill.aiSkillType);
  ptr = parse(ptr, skill.params[0]);
  ptr = parse(ptr, skill.params[1]);
  ptr = parse(ptr, skill.params[2]);
  ptr = parse(ptr, skill.params[3]);
  ptr = parse(ptr, skill.params[4]);
  ptr = parse(ptr, skill.params[5]);
  ptr = parse(ptr, skill.params[6]);
  ptr = parse(ptr, skill.params[7]);
  ptr = parse(ptr, skill.params[8]);
  ptr = parse(ptr, skill.params[9]);
  ptr = parse(ptr, skill.params[10]);
  ptr = parse(ptr, skill.params[11]);
  ptr = parse(ptr, skill.params[12]);
  ptr = parse(ptr, skill.params[13]);
  ptr = parse(ptr, skill.params[14]);
  ptr = parse(ptr, skill.params[15]);
  ptr = parse(ptr, skill.params[16]);
  ptr = parse(ptr, skill.params[17]);
  ptr = parse(ptr, skill.params[18]);
  ptr = parse(ptr, skill.params[19]);
  ptr = parse(ptr, skill.params[20]);
  ptr = parse(ptr, skill.params[21]);
  ptr = parse(ptr, skill.params[22]);
  ptr = parse(ptr, skill.params[23]);
  ptr = parse(ptr, skill.params[24]);
  ptr = parse(ptr, skill.params[25]);
  ptr = parse(ptr, skill.params[26]);
  ptr = parse(ptr, skill.params[27]);
  ptr = parse(ptr, skill.params[28]);
  ptr = parse(ptr, skill.params[29]);
  ptr = parse(ptr, skill.params[30]);
  ptr = parse(ptr, skill.params[31]);
  ptr = parse(ptr, skill.params[32]);
  ptr = parse(ptr, skill.params[33]);
  ptr = parse(ptr, skill.params[34]);
  ptr = parse(ptr, skill.params[35]);
  ptr = parse(ptr, skill.params[36]);
  ptr = parse(ptr, skill.params[37]);
  ptr = parse(ptr, skill.params[38]);
  ptr = parse(ptr, skill.params[39]);
  ptr = parse(ptr, skill.params[40]);
  ptr = parse(ptr, skill.params[41]);
  ptr = parse(ptr, skill.params[42]);
  ptr = parse(ptr, skill.params[43]);
  ptr = parse(ptr, skill.params[44]);
  ptr = parse(ptr, skill.params[45]);
  ptr = parse(ptr, skill.params[46]);
  ptr = parse(ptr, skill.params[47]);
  ptr = parse(ptr, skill.params[48]);
  parse(ptr, skill.params[49]);
  return skill;
}

pk2::ref::Teleport parseTeleportbuildingLine(const std::string &line) {
	pk2::ref::Teleport teleportBuilding;
  const char *ptr = line.data();
  ptr = parse(ptr, teleportBuilding.service);
  ptr = parse(ptr, teleportBuilding.id);
  ptr = parse(ptr, teleportBuilding.codeName128);
  ptr = parse(ptr, teleportBuilding.objName128);
  ptr = parse(ptr, teleportBuilding.orgObjCodeName128);
  ptr = parse(ptr, teleportBuilding.nameStrID128);
  ptr = parse(ptr, teleportBuilding.descStrID128);
  ptr = parse(ptr, teleportBuilding.cashItem);
  ptr = parse(ptr, teleportBuilding.bionic);
  ptr = parse(ptr, teleportBuilding.typeId1);
  ptr = parse(ptr, teleportBuilding.typeId2);
  ptr = parse(ptr, teleportBuilding.typeId3);
  ptr = parse(ptr, teleportBuilding.typeId4);
  ptr = parse(ptr, teleportBuilding.decayTime);
  ptr = parse(ptr, teleportBuilding.country);
  ptr = parse(ptr, teleportBuilding.rarity);
  ptr = parse(ptr, teleportBuilding.canTrade);
  ptr = parse(ptr, teleportBuilding.canSell);
  ptr = parse(ptr, teleportBuilding.canBuy);
  ptr = parse(ptr, teleportBuilding.canBorrow);
  ptr = parse(ptr, teleportBuilding.canDrop);
  ptr = parse(ptr, teleportBuilding.canPick);
  ptr = parse(ptr, teleportBuilding.canRepair);
  ptr = parse(ptr, teleportBuilding.canRevive);
  ptr = parse(ptr, teleportBuilding.canUse);
  ptr = parse(ptr, teleportBuilding.canThrow);
  ptr = parse(ptr, teleportBuilding.price);
  ptr = parse(ptr, teleportBuilding.costRepair);
  ptr = parse(ptr, teleportBuilding.costRevive);
  ptr = parse(ptr, teleportBuilding.costBorrow);
  ptr = parse(ptr, teleportBuilding.keepingFee);
  ptr = parse(ptr, teleportBuilding.sellPrice);
  ptr = parse(ptr, teleportBuilding.reqLevelType1);
  ptr = parse(ptr, teleportBuilding.reqLevel1);
  ptr = parse(ptr, teleportBuilding.reqLevelType2);
  ptr = parse(ptr, teleportBuilding.reqLevel2);
  ptr = parse(ptr, teleportBuilding.reqLevelType3);
  ptr = parse(ptr, teleportBuilding.reqLevel3);
  ptr = parse(ptr, teleportBuilding.reqLevelType4);
  ptr = parse(ptr, teleportBuilding.reqLevel4);
  ptr = parse(ptr, teleportBuilding.maxContain);
  ptr = parse(ptr, teleportBuilding.regionID);
  ptr = parse(ptr, teleportBuilding.dir);
  ptr = parse(ptr, teleportBuilding.offsetX);
  ptr = parse(ptr, teleportBuilding.offsetY);
  ptr = parse(ptr, teleportBuilding.offsetZ);
  ptr = parse(ptr, teleportBuilding.speed1);
  ptr = parse(ptr, teleportBuilding.speed2);
  ptr = parse(ptr, teleportBuilding.scale);
  ptr = parse(ptr, teleportBuilding.bcHeight);
  ptr = parse(ptr, teleportBuilding.bcRadius);
  ptr = parse(ptr, teleportBuilding.eventID);
  ptr = parse(ptr, teleportBuilding.assocFileObj128);
  ptr = parse(ptr, teleportBuilding.assocFileDrop128);
  ptr = parse(ptr, teleportBuilding.assocFileIcon128);
  ptr = parse(ptr, teleportBuilding.assocFile1_128);
  ptr = parse(ptr, teleportBuilding.assocFile2_128);
  parse(ptr, teleportBuilding.link);
  return teleportBuilding;
}

pk2::ref::ScrapOfPackageItem parseScrapOfPackageItemLine(const std::string &line) {
	pk2::ref::ScrapOfPackageItem scrap;
  const char *ptr = line.data();
  ptr = parse(ptr, scrap.service);
  ptr = parse(ptr, scrap.country);
  ptr = parse(ptr, scrap.refPackageItemCodeName);
  ptr = parse(ptr, scrap.refItemCodeName);
  ptr = parse(ptr, scrap.optLevel);
  ptr = parse(ptr, scrap.variance);
  ptr = parse(ptr, scrap.data);
  ptr = parse(ptr, scrap.magParamNum);
  ptr = parse(ptr, scrap.magParams[0]);
  ptr = parse(ptr, scrap.magParams[1]);
  ptr = parse(ptr, scrap.magParams[2]);
  ptr = parse(ptr, scrap.magParams[3]);
  ptr = parse(ptr, scrap.magParams[4]);
  ptr = parse(ptr, scrap.magParams[5]);
  ptr = parse(ptr, scrap.magParams[6]);
  ptr = parse(ptr, scrap.magParams[7]);
  ptr = parse(ptr, scrap.magParams[8]);
  ptr = parse(ptr, scrap.magParams[9]);
  ptr = parse(ptr, scrap.magParams[10]);
  ptr = parse(ptr, scrap.magParams[11]);
  ptr = parse(ptr, scrap.param1);
  ptr = parse(ptr, scrap.param1Desc128);
  ptr = parse(ptr, scrap.param2);
  ptr = parse(ptr, scrap.param2Desc128);
  ptr = parse(ptr, scrap.param3);
  ptr = parse(ptr, scrap.param3Desc128);
  ptr = parse(ptr, scrap.param4);
  ptr = parse(ptr, scrap.param4Desc128);
  parse(ptr, scrap.index);
	return scrap;
}

pk2::ref::ShopTab parseShopTabLine(const std::string &line) {
	pk2::ref::ShopTab tab;
  const char *ptr = line.data();
  ptr = parse(ptr, tab.service);
  ptr = parse(ptr, tab.country);
	ptr = parse(ptr, tab.id);
  ptr = parse(ptr, tab.codeName128);
  ptr = parse(ptr, tab.refTabGroupCodeName);
  parse(ptr, tab.strID128Tab);
	return tab;
}

pk2::ref::ShopGroup parseShopGroupLine(const std::string &line) {
	pk2::ref::ShopGroup group;
  const char *ptr = line.data();
  ptr = parse(ptr, group.service);
  ptr = parse(ptr, group.country);
  ptr = parse(ptr, group.id);
  ptr = parse(ptr, group.codeName128);
  ptr = parse(ptr, group.refNPCCodeName);
  ptr = parse(ptr, group.param1);
  ptr = parse(ptr, group.param1Desc128);
  ptr = parse(ptr, group.param2);
  ptr = parse(ptr, group.param2Desc128);
  ptr = parse(ptr, group.param3);
  ptr = parse(ptr, group.param3Desc128);
  ptr = parse(ptr, group.param4);
  parse(ptr, group.param4Desc128);
  return group;
}

pk2::ref::ShopGood parseShopGoodLine(const std::string &line) {
	pk2::ref::ShopGood good;
  const char *ptr = line.data();
  ptr = parse(ptr, good.service);
  ptr = parse(ptr, good.country);
  ptr = parse(ptr, good.refTabCodeName);
  ptr = parse(ptr, good.refPackageItemCodeName);
  ptr = parse(ptr, good.slotIndex);
  ptr = parse(ptr, good.param1);
  ptr = parse(ptr, good.param1Desc128);
  ptr = parse(ptr, good.param2);
  ptr = parse(ptr, good.param2Desc128);
  ptr = parse(ptr, good.param3);
  ptr = parse(ptr, good.param3Desc128);
  ptr = parse(ptr, good.param4);
  parse(ptr, good.param4Desc128);
	return good;
}

pk2::ref::MappingShopGroup parseMappingShopGroupLine(const std::string &line) {
	pk2::ref::MappingShopGroup mapping;
  const char *ptr = line.data();
  ptr = parse(ptr, mapping.service);
  ptr = parse(ptr, mapping.country);
  ptr = parse(ptr, mapping.refShopGroupCodeName);
  parse(ptr, mapping.refShopCodeName);
	return mapping;
}

pk2::ref::MappingShopWithTab parseMappingShopWithTabLine(const std::string &line) {
	pk2::ref::MappingShopWithTab mapping;
  const char *ptr = line.data();
  ptr = parse(ptr, mapping.service);
  ptr = parse(ptr, mapping.country);
  ptr = parse(ptr, mapping.refShopCodeName);
  parse(ptr, mapping.refTabGroupCodeName);
	return mapping;
}

DivisionInfo parseDivisionInfo(const std::vector<uint8_t> &data) {
	DivisionInfo divisionInfo;
	int readIndex=0;
	divisionInfo.locale = get<uint8_t>(data, readIndex);
	auto divisionCount = get<uint8_t>(data, readIndex);
	for (int i=0; i<divisionCount; ++i) {
		Division division;
		auto divisionNameLength = get<uint32_t>(data, readIndex);
		for (int j=0; j<divisionNameLength; ++j) {
			char c = get<char>(data, readIndex);
			division.name += c;
		}
		++readIndex; // throw away null terminator
		auto gatewayCount = get<uint8_t>(data, readIndex);
		for (int j=0; j<gatewayCount; ++j) {
			auto addressLength = get<uint32_t>(data, readIndex);
			std::string address;
			for (int k=0; k<addressLength; ++k) {
				char c = get<char>(data, readIndex);
				address += c;
			}
			++readIndex; // throw away null terminator
			division.gatewayIpAddresses.push_back(address);
			divisionInfo.divisions.push_back(division);
		}
	}
	return divisionInfo;
}

std::vector<std::string> split(const std::string &str, const std::string &delim) {
	std::vector<std::string> result;
	size_t last = 0;
	size_t next = 0;
	while ((next = str.find(delim, last)) != std::string::npos) {
		std::string s = str.substr(last, next-last);
		if (s != "") {
			result.push_back(s);
		}
		last = next + delim.size();
	}
	std::string s = str.substr(last);
	if (s != "") {
		result.push_back(s);
	}
	return result;		
}

std::vector<std::string> splitAndSelectFields(const std::string &str, const std::string &delim, const std::vector<int> &fields) {
	if (fields.empty()) {
		return {};
	}
	int fieldsIndex=0;
	std::vector<std::string> result;
	size_t last = 0;
	size_t next = 0;
	int fieldNum=0;
	while ((next = str.find(delim, last)) != std::string::npos) {
		std::string s = str.substr(last, next-last);
		if (s != "") {
			if (fields[fieldsIndex] == fieldNum) {
				result.push_back(s);
				++fieldsIndex;
				if (fieldsIndex >= fields.size()) {
					return result;
				}
			}
		}
		last = next + delim.size();
		++fieldNum;
	}
	std::string s = str.substr(last);
	if (s != "") {
		if (fields[fieldsIndex] == fieldNum) {
			result.push_back(s);
		}
	}
	return result;		
}

namespace {
void copySubstring(const char *begin, char **end, std::string &dest) {
  const char *ptr = begin;
  while (*ptr != 0 && *ptr != '\t') {
    ++ptr;
  }
  std::string_view sv(begin, ptr-begin);
  dest = sv;
  *end = const_cast<char*>(ptr);
}
} // namespace

template<>
const char* parse<std::string>(const char *begin, std::string &result) {
  char *end;
  copySubstring(begin, &end, result);
  return end+1;
}

template<>
const char* parse<uint8_t>(const char *begin, uint8_t &result) {
  char *end;
  result = strtol(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<int16_t>(const char *begin, int16_t &result) {
  char *end;
  result = strtol(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<int32_t>(const char *begin, int32_t &result) {
  char *end;
  result = strtol(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<int64_t>(const char *begin, int64_t &result) {
  char *end;
  result = strtoll(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<float>(const char *begin, float &result) {
  char *end;
  result = strtof(begin, &end);
  return end+1;
}

} // namespace pk2::parsing