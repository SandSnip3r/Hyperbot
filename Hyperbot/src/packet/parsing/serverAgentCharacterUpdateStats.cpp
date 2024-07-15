#include "serverAgentCharacterUpdateStats.hpp"

namespace packet::parsing {

ServerAgentCharacterUpdateStats::ServerAgentCharacterUpdateStats(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint32_t phyAtkMin = stream.Read<uint32_t>();
  uint32_t phyAtkMax = stream.Read<uint32_t>();
  uint32_t magAtkMin = stream.Read<uint32_t>();
  uint32_t magAtkMax = stream.Read<uint32_t>();
  uint16_t phyDef = stream.Read<uint16_t>();
  uint16_t magDef = stream.Read<uint16_t>();
  uint16_t hitRate = stream.Read<uint16_t>();
  uint16_t parryRate = stream.Read<uint16_t>();
  stream.Read(maxHp_);
  stream.Read(maxMp_);
  stream.Read(strPoints_);
  stream.Read(intPoints_);
}

} // namespace packet::parsing