#ifndef PK2_REF_TEXT_ZONE_NAME_HPP_
#define PK2_REF_TEXT_ZONE_NAME_HPP_

#include <cstdint>
#include <string>
#include <ostream>

namespace sro::pk2::ref {

struct TextZoneName {
  int32_t service;
  std::string codeName128;
  std::string korean;
  std::string unkLang0;
  std::string chineseTraditional;
  std::string chineseSimplified;
  std::string german;
  std::string japanese;
  std::string english;
  std::string vietnamese;
  std::string portuguese;
  std::string russian;
  std::string turkish;
  std::string spanish;
  std::string arabic;
};

std::ostream& operator<<(std::ostream &stream, const TextZoneName &zoneName);

} // namespace sro::pk2::ref

#endif // PK2_REF_TEXT_ZONE_NAME_HPP_