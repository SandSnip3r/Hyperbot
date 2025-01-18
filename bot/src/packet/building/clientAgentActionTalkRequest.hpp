#include "packet/opcode.hpp"
#include "packet/enums/packetEnums.hpp"

#include "shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_ACTION_TALK_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_ACTION_TALK_REQUEST_HPP

namespace packet::building {

class ClientAgentActionTalkRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentActionTalkRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint32_t gId, packet::enums::TalkOption talkOption);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_ACTION_TALK_REQUEST_HPP