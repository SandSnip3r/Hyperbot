#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATUS_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATUS_HPP

#include "packet/parsing/parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <array>
#include <cstdint>

namespace packet::parsing {

class ServerAgentEntityUpdateStatus : public ParsedPacket {
public:
  using ModernStateLevelArrayType = std::array<uint8_t, 32>;
  ServerAgentEntityUpdateStatus(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId globalId() const;
  enums::UpdateFlag updateFlag() const;
  enums::VitalInfoFlag vitalBitmask() const;
  uint32_t newHpValue() const;
  uint32_t newMpValue() const;
  uint16_t newHgpValue() const;
  uint32_t stateBitmask() const;
  // Note that the indices for legacy states will always be 0.
  const ModernStateLevelArrayType& modernStateLevels() const;
private:
  sro::scalar_types::EntityGlobalId entityUniqueId_;
  enums::UpdateFlag updateFlag_;
  enums::VitalInfoFlag vitalBitmask_;
  uint32_t newHpValue_;
  uint32_t newMpValue_;
  uint16_t newHgpValue_;
  uint32_t stateBitmask_;
  ModernStateLevelArrayType modernStateLevels_{};
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATUS_HPP