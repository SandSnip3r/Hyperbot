#ifndef PACKET_PARSING_SEVER_AGENT_ACTION_SELECT_RESPONSE_HPP
#define PACKET_PARSING_SEVER_AGENT_ACTION_SELECT_RESPONSE_HPP

#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/parsedPacket.hpp"
#include "state/entityTracker.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>

namespace packet::parsing {

class ServerAgentActionSelectResponse : public ParsedPacket {
public:
  ServerAgentActionSelectResponse(const PacketContainer &packet, const state::EntityTracker &entityTracker);
  uint8_t result() const { return result_; }
  uint16_t errorCode() const { return errorCode_; }
  sro::scalar_types::EntityGlobalId globalId() const { return globalId_; }
  enums::VitalInfoFlag vitalInfoMask() const { return vitalInfoMask_; }
  uint32_t hp() const { return hp_; }
private:
  uint8_t result_;
  uint16_t errorCode_;
  sro::scalar_types::EntityGlobalId globalId_;
  enums::VitalInfoFlag vitalInfoMask_;
  uint32_t hp_; //a
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SEVER_AGENT_ACTION_SELECT_RESPONSE_HPP