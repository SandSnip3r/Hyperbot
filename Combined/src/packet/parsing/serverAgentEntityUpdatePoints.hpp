#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POINTS_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POINTS_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"

#include <cstdint>

namespace packet::parsing {
  
class ServerAgentEntityUpdatePoints : public ParsedPacket {
public:
  ServerAgentEntityUpdatePoints(const PacketContainer &packet);
  packet::enums::UpdatePointsType updatePointsType() const;
  uint64_t gold() const;
  uint32_t skillPoints() const;
  bool isDisplayed() const;
private:
  packet::enums::UpdatePointsType updatePointsType_;
  uint64_t gold_;
  uint32_t skillPoints_;
  bool isDisplayed_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POINTS_HPP