#ifndef PK2_MEDIA_TELEPORT_HPP_
#define PK2_MEDIA_TELEPORT_HPP_

#include <cstdint>
#include <string>

namespace pk2::media {

using TeleportId = uint32_t;

struct Teleport {
	TeleportId id;
  std::string codeName128;
  uint8_t typeId1;
  uint8_t typeId2;
  uint8_t typeId3;
  uint8_t typeId4;
};

} // namespace pk2::media

#endif // PK2_MEDIA_TELEPORT_HPP_