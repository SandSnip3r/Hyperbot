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
  std::shared_ptr<entity::Entity> entity = entityTracker.getEntity(globalId_);
  if (entity->typeId1 == 1) { // TODO: Move to new type id categories
    // Character
    vitalInfoMask_ = stream.Read<enums::VitalInfoFlag>();
    if (flags::isSet(vitalInfoMask_, enums::VitalInfoFlag::kVitalInfoHp)) {
      hp_ = stream.Read<uint32_t>();
    }
    // TODO: Talk options follow, don't care for now
  } // TODO: else if (entity->typeId1 == 4) { // Talk options
}
// a
} // namespace packet::parsing