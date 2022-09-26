#include "serverAgentEntityUpdateState.hpp"

namespace packet::parsing {

ServerAgentEntityUpdateState::ServerAgentEntityUpdateState(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<sro::scalar_types::EntityGlobalId>();
  stateType_ = stream.Read<StateType>();
  state_ = stream.Read<uint8_t>();
  if (stateType_ == StateType::kBodyState) {
    if (static_cast<enums::BodyState>(state_) == enums::BodyState::kHwan) {
      isEnhanced_ = stream.Read<uint8_t>();
    }
  }
}

sro::scalar_types::EntityGlobalId ServerAgentEntityUpdateState::globalId() const {
  return globalId_;
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