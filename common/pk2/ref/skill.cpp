#include "skill.hpp"

#include <algorithm>

namespace pk2::ref {
  
bool Skill::isEfta() const {
  const int32_t kEfta = 1701213281;
  for (const auto param : params) {
    if (param == kEfta) {
      return true;
    }
  }
  return false;
}

namespace {

uint32_t getVal(const std::string &str) {
  uint32_t val = 0;
  for (int i=0; i<4 && i<str.size(); ++i) {
    val <<= 8;
    val |= static_cast<uint8_t>(str[i]);
  }
  return val;
}

const uint32_t kReqiVal = getVal("reqi");
const uint32_t kTel2Val = getVal("tel2");
const uint32_t kTel3Val = getVal("tel3");
const uint32_t kTeleVal = getVal("tele");
const uint32_t kDuraVal = getVal("dura");

}

std::vector<RequiredWeapon> Skill::reqi() const {
  std::vector<RequiredWeapon> result;
  for (int paramNum=0; paramNum<params.size(); ++paramNum) {
    if (params[paramNum] == kReqiVal) {
      result.emplace_back(RequiredWeapon{static_cast<uint8_t>(params[paramNum+1]), static_cast<uint8_t>(params[paramNum+2])});
      paramNum += 2;
    }
  }
  return result;
}

bool Skill::isInstant() const {
  // TODO: Do we need to check chain?
  return basicActivity == 1;
  // return basicChainCode == 0 && (actionPreparingTime+actionCastingTime+actionActionDuration) == 0;
}

bool Skill::isTele() const {
  return (std::find_if(params.begin(), params.end(), [](const int32_t &param){
    return (param == kTel2Val || param == kTel3Val || param == kTeleVal);
  }) != params.end());
}

bool Skill::isPseudoinstant() const {
  return (basicActivity != 1 &&
          (actionPreparingTime+actionCastingTime+actionActionDuration) == 0 &&
          std::find(params.begin(), params.end(), kDuraVal) != params.end());
}

} // namespace pk2::ref
