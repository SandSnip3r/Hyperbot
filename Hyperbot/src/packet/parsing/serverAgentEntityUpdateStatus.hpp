#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATUS_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATUS_HPP

#include "packet/parsing/parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <vector>

namespace packet::parsing {

class ServerAgentEntityUpdateStatus : public ParsedPacket {
public:
  ServerAgentEntityUpdateStatus(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId entityUniqueId() const;
  enums::UpdateFlag updateFlag() const;
  enums::VitalInfoFlag vitalBitmask() const;
  uint32_t newHpValue() const;
  uint32_t newMpValue() const;
  uint16_t newHgpValue() const;
  uint32_t stateBitmask() const;
  const std::vector<uint8_t>& stateLevels() const;
private:
  sro::scalar_types::EntityGlobalId entityUniqueId_;
  enums::UpdateFlag updateFlag_;
  enums::VitalInfoFlag vitalBitmask_;
  uint32_t newHpValue_;
  uint32_t newMpValue_;
  uint16_t newHgpValue_;
  uint32_t stateBitmask_;
  std::vector<uint8_t> stateLevels_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATUS_HPP