#ifndef PK2_MEDIA_SKILL_HPP_
#define PK2_MEDIA_SKILL_HPP_

#include <array>
#include <string>

namespace pk2::ref {

using SkillId = int32_t;

struct Skill {
  uint8_t service;
  int32_t id;
  int32_t groupId;
  std::string basicCode;
  std::string basicName;
  std::string basicGroup;
  int32_t basicOriginal;
  uint8_t basicLevel;
  uint8_t basicActivity;
  int32_t basicChainCode;
  int32_t basicRecycleCost;
  int32_t actionPreparingTime;
  int32_t actionCastingTime;
  int32_t actionActionDuration;
  int32_t actionReuseDelay;
  int32_t actionCoolTime;
  int32_t actionFlyingSpeed;
  uint8_t actionInterruptable;
  int32_t actionOverlap;
  uint8_t actionAutoAttackType;
  uint8_t actionInTown;
  int16_t actionRange;
  uint8_t targetRequired;
  uint8_t targetTypeAnimal;
  uint8_t targetTypeLand;
  uint8_t targetTypeBuilding;
  uint8_t targetGroupSelf;
  uint8_t targetGroupAlly;
  uint8_t targetGroupParty;
  uint8_t targetGroupEnemy_M;
  uint8_t targetGroupEnemy_P;
  uint8_t targetGroupNeutral;
  uint8_t targetGroupDontCare;
  uint8_t targetEtcSelectDeadBody;
  int32_t reqCommonMastery1;
  int32_t reqCommonMastery2;
  uint8_t reqCommonMasteryLevel1;
  uint8_t reqCommonMasteryLevel2;
  int16_t reqCommonStr;
  int16_t reqCommonInt;
  int32_t reqLearnSkill1;
  int32_t reqLearnSkill2;
  int32_t reqLearnSkill3;
  uint8_t reqLearnSkillLevel1;
  uint8_t reqLearnSkillLevel2;
  uint8_t reqLearnSkillLevel3;
  int32_t reqLearnSP;
  uint8_t reqLearnRace;
  uint8_t reqRestriction1;
  uint8_t reqRestriction2;
  uint8_t reqCastWeapon1;
  uint8_t reqCastWeapon2;
  int16_t consumeHP;
  int16_t consumeMP;
  int16_t consumeHPRatio;
  int16_t consumeMPRatio;
  uint8_t consumeWHAN;
  uint8_t uiSkillTab;
  uint8_t uiSkillPage;
  uint8_t uiSkillColumn;
  uint8_t uiSkillRow;
  std::string uiIconFile;
  std::string uiSkillName;
  std::string uiSkillToolTip;
  std::string uiSkillToolTip_Desc;
  std::string uiSkillStudy_Desc;
  int16_t aiAttackChance;
  uint8_t aiSkillType;
  std::array<int32_t,50> params;
  // int32_t param1 // 68
  // int32_t param2 // 69
  // int32_t param3 // 70
  // int32_t param4 // 71
  // int32_t param5 // 72
  // int32_t param6 // 73
  // int32_t param7 // 74
  // int32_t param8 // 75
  // int32_t param9 // 76
  // int32_t param10 // 77
  // int32_t param11 // 78
  // int32_t param12 // 79
  // int32_t param13 // 80
  // int32_t param14 // 81
  // int32_t param15 // 82
  // int32_t param16 // 83
  // int32_t param17 // 84
  // int32_t param18 // 85
  // int32_t param19 // 86
  // int32_t param20 // 87
  // int32_t param21 // 88
  // int32_t param22 // 89
  // int32_t param23 // 90
  // int32_t param24 // 91
  // int32_t param25 // 92
  // int32_t param26 // 93
  // int32_t param27 // 94
  // int32_t param28 // 95
  // int32_t param29 // 96
  // int32_t param30 // 97
  // int32_t param31 // 98
  // int32_t param32 // 99
  // int32_t param33 // 100
  // int32_t param34 // 101
  // int32_t param35 // 102
  // int32_t param36 // 103
  // int32_t param37 // 104
  // int32_t param38 // 105
  // int32_t param39 // 106
  // int32_t param40 // 107
  // int32_t param41 // 108
  // int32_t param42 // 109
  // int32_t param43 // 110
  // int32_t param44 // 111
  // int32_t param45 // 112
  // int32_t param46 // 113
  // int32_t param47 // 114
  // int32_t param48 // 115
  // int32_t param49 // 116
  // int32_t param50 // 117
	// SkillId id;
	// std::string basicCode;
  // std::string basicName;
  // std::string basicGroup;
  // std::array<int32_t,50> params;
  
  // Efta or atfe ("auto transfer effect") like Recovery Division or bard dances
  bool isEfta() const;
};

} // namespace pk2::ref

#endif // PK2_MEDIA_SKILL_HPP_