#include "serverAgentActionCommandResponse.hpp"

namespace packet::parsing {

ServerAgentActionCommandResponse::ServerAgentActionCommandResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  actionState_ = static_cast<enums::ActionState>(stream.Read<uint8_t>());
  repeatAction_ = stream.Read<uint8_t>();
  if (actionState_ == enums::ActionState::kError) {
    errorCode_ = stream.Read<uint16_t>(); 
  }
}

enums::ActionState ServerAgentActionCommandResponse::actionState() const {
  return actionState_;
}

bool ServerAgentActionCommandResponse::repeatAction() const {
  return repeatAction_;
}

uint16_t ServerAgentActionCommandResponse::errorCode() const {
  // Skill on cooldown
  //0x4004
  
  //Unable to use that skill at this moment
  //UIIT_SKILL_USE_FAIL_SEALED = 0x0140, 0x0240

  //[Not displayed] -> 0xB070 overrides.
  //UIIT_SKILL_USE_FAIL_WRONGWEAPON = 0x0440

  //When a transport is summoned, you cannot use the follow function.
  //UIIT_MSG_COS_BAN_FOLLOW_COS = 0x0540
  return errorCode_;
}

} // namespace packet::parsing