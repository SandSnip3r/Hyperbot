#ifndef PK2_REF_MAGIC_OPTION_HPP_
#define PK2_REF_MAGIC_OPTION_HPP_

#include <cstdint>
#include <string>
#include <ostream>

namespace sro::pk2::ref {

using MagicOptionId = int16_t;

struct MagicOption {
  int32_t service;
  int16_t id;
  std::string mOptName128;
  std::string attrType;
  int32_t mLevel;
  float prob;
  int32_t weight;
  int32_t param1;
  int32_t param2;
  int32_t param3;
  int32_t param4;
  int32_t param5;
  int32_t param6;
  int32_t param7;
  int32_t param8;
  int32_t param9;
  int32_t param10;
  int32_t param11;
  int32_t param12;
  int32_t param13;
  int32_t param14;
  int32_t param15;
  int32_t param16;
  int32_t excFunc1;
  int32_t excFunc2;
  int32_t excFunc3;
  int32_t excFunc4;
  int32_t excFunc5;
  int32_t excFunc6;
  std::string availItemGroup1;
  int32_t reqClass1;
  std::string availItemGroup2;
  int32_t reqClass2;
  std::string availItemGroup3;
  int32_t reqClass3;
  std::string availItemGroup4;
  int32_t reqClass4;
  std::string availItemGroup5;
  int32_t reqClass5;
  std::string availItemGroup6;
  int32_t reqClass6;
  std::string availItemGroup7;
  int32_t reqClass7;
  std::string availItemGroup8;
  int32_t reqClass8;
  std::string availItemGroup9;
  int32_t reqClass9;
  std::string availItemGroup10;
  int32_t reqClass10;
};

std::ostream& operator<<(std::ostream &stream, const MagicOption &magOpt);

} // namespace sro::pk2::ref

#endif // PK2_REF_MAGIC_OPTION_HPP_