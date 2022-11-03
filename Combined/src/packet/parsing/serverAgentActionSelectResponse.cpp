#include "logging.hpp"

#include "serverAgentActionSelectResponse.hpp"

namespace packet::parsing {

ServerAgentActionSelectResponse::ServerAgentActionSelectResponse(const PacketContainer &packet, const state::EntityTracker &entityTracker) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  if (result_ == 2) {
    // Code 4 for items, or npc/player too far (but still in view)
    errorCode_ = stream.Read<uint16_t>();
    return;
  }

  // Success
  globalId_ = stream.Read<sro::scalar_types::EntityGlobalId>();
  const auto *entity = entityTracker.getEntity(globalId_);
  if (!entity) {
    throw std::runtime_error("Selected entity that we're not tracking");
  }
  if (entity->typeId1 == 1) { // TODO: Move to new type id categories
    // Character
    vitalInfoMask_ = stream.Read<enums::VitalInfoFlag>();
    if (flags::isSet(vitalInfoMask_, enums::VitalInfoFlag::kVitalInfoHp)) {
      hp_ = stream.Read<uint32_t>();
    }
    // TODO: Talk options follow, don't care for now
  } // TODO: else if (entity->typeId1 == 4) { // Talk options
}

uint8_t ServerAgentActionSelectResponse::result() const {
  return result_;
}

uint16_t ServerAgentActionSelectResponse::errorCode() const {
  return errorCode_;
}

sro::scalar_types::EntityGlobalId ServerAgentActionSelectResponse::globalId() const {
  return globalId_;
}

enums::VitalInfoFlag ServerAgentActionSelectResponse::vitalInfoMask() const {
  return vitalInfoMask_;
}

uint32_t ServerAgentActionSelectResponse::hp() const {
  return hp_;
}

} // namespace packet::parsing