#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_MOVEMENT_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_MOVEMENT_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace packet::parsing {
  
class ServerAgentEntityUpdateMovement : public ParsedPacket {
public:
  ServerAgentEntityUpdateMovement(const PacketContainer &packet);
  uint32_t globalId() const;
  bool hasDestination() const;
  bool hasSource() const;
  const packet::structures::Position& destinationPosition() const;
  packet::enums::AngleAction angleAction() const;
  uint16_t angle() const;
  const packet::structures::Position& sourcePosition() const;
private:
  uint32_t globalId_;
  std::optional<packet::structures::Position> destinationPosition_;
  packet::enums::AngleAction angleAction_;
  uint16_t angle_;
  std::optional<packet::structures::Position> sourcePosition_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_MOVEMENT_HPP