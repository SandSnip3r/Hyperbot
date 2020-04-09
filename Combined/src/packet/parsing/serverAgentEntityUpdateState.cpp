#include "serverAgentEntityUpdateState.hpp"

namespace packet::parsing {

ServerAgentEntityUpdateState::ServerAgentEntityUpdateState(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  gId_ = stream.Read<uint32_t>();
  stateType_ = static_cast<StateType>(stream.Read<uint8_t>());
  state_ = stream.Read<uint8_t>();
  if (stateType_ == StateType::kBodyState) {
    if (static_cast<enums::BodyState>(state_) == enums::BodyState::kHwan) {
      isEnhanced_ = stream.Read<uint8_t>();
    }
  }
}

uint32_t ServerAgentEntityUpdateState::gId() const {
  return gId_;
}

StateType ServerAgentEntityUpdateState::stateType() const {
  return stateType_;
}

uint8_t ServerAgentEntityUpdateState::state() const {
  return state_;
}

bool ServerAgentEntityUpdateState::isEnhanced() const {
  return isEnhanced_;
}

} // namespace packet::parsing