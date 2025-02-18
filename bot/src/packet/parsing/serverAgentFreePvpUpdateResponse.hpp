#ifndef PACKET_PARSING_SERVER_AGENT_FREE_PVP_UPDATE_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_FREE_PVP_UPDATE_RESPONSE_HPP_

#include "packet/parsing/parsedPacket.hpp"

#include <silkroad_lib/scalar_types.hpp>

namespace packet::parsing {

class ServerAgentFreePvpUpdateResponse : public ParsedPacket {
public:
  ServerAgentFreePvpUpdateResponse(const PacketContainer &packet);
  uint8_t result() const { return result_; }
  sro::scalar_types::EntityGlobalId globalId() const { return globalId_; }
  packet::enums::FreePvpMode mode() const { return mode_; }
  uint16_t errorCode() const { return errorCode_; }
private:
  uint8_t result_;
  sro::scalar_types::EntityGlobalId globalId_;
  packet::enums::FreePvpMode mode_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_FREE_PVP_UPDATE_RESPONSE_HPP_