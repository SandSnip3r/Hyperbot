#include "skill.hpp"

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

}

std::vector<std::pair<uint8_t, uint8_t>> Skill::reqi() const {
  std::vector<std::pair<uint8_t, uint8_t>> result;
  for (int paramNum=0; paramNum<params.size(); ++paramNum) {
    if (params[paramNum] == kReqiVal) {
      result.emplace_back(params[paramNum+1], params[paramNum+2]);
      paramNum += 2;
    }
  }
  return result;
}

} // namespace pk2::ref
