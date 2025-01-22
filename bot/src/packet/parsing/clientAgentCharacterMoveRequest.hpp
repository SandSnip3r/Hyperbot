#ifndef PACKET_PARSING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP
#define PACKET_PARSING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <silkroad_lib/position.hpp>

#include <cstdint>
#include <optional>

namespace packet::parsing {

class ClientAgentCharacterMoveRequest : public ParsedPacket {
public:
  ClientAgentCharacterMoveRequest(const PacketContainer &packet);
  bool hasDestination() const;
  const sro::Position& destinationPosition() const;
  packet::enums::AngleAction angleAction() const;
  uint16_t angle() const;
private:
  std::optional<sro::Position> destinationPosition_;
  packet::enums::AngleAction angleAction_;
  uint16_t angle_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_CHARACTER_MOVE_REQUEST_HPP