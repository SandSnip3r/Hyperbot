#ifndef PK2_MEDIA_CHARACTER_HPP_
#define PK2_MEDIA_CHARACTER_HPP_

#include <string>
#include <ostream>

namespace pk2::media {

using CharacterId = uint32_t;

struct Character {
	CharacterId id;
  std::string codeName128;
  uint8_t country;
  uint8_t charGender;
};

std::ostream& operator<<(std::ostream &stream, const Character &character);

} // namespace pk2::media

#endif // PK2_MEDIA_CHARACTER_HPP_