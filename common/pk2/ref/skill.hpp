#ifndef PK2_MEDIA_SKILL_HPP_
#define PK2_MEDIA_SKILL_HPP_

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pk2::ref {

namespace skill_param {

extern const int32_t kHaste;

} // namespace skill_param

struct RequiredWeapon {
  uint8_t typeId3, typeId4;
};

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
  std::array<int32_t,50> params = {0};
  
  // Efta or atfe ("auto transfer effect") like Recovery Division or bard dances
  bool isEfta() const;
  bool isImbue() const;
  bool hasParam(int32_t param) const;
  std::vector<RequiredWeapon> reqi() const;
  bool isInstant() const;
  bool isTele() const;
  bool isPseudoinstant() const;
  int32_t duration() const;

  enum class Param1Type {
    kMelee = 0,
    kRanged = 1,
    // 2 is unknown
    kBuff = 3,
    kPassive = 4
  };

  Param1Type param1Type() const;
};

} // namespace pk2::ref

#endif // PK2_MEDIA_SKILL_HPP_