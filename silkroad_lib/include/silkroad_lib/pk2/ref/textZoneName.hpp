#ifndef PK2_REF_TEXT_ZONE_NAME_HPP_
#define PK2_REF_TEXT_ZONE_NAME_HPP_

#include <cstdint>
#include <string>
#include <ostream>

namespace sro::pk2::ref {

struct TextZoneName {
  int32_t service;
  std::string key;
  std::string korean;
  std::string unkLang0;
  std::string unkLang1;
  std::string unkLang2;
  std::string unkLang3;
  std::string unkLang4;
  std::string english;
  std::string vietnamese;
  std::string unkLang5;
  std::string unkLang6;
  std::string unkLang7;
  std::string unkLang8;
  std::string unkLang9;
  std::string unkLang10;
};

std::ostream& operator<<(std::ostream &stream, const TextZoneName &zoneName);

} // namespace sro::pk2::ref

#endif // PK2_REF_TEXT_ZONE_NAME_HPP_