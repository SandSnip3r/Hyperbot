#include "packet/parsing/commonParsing.hpp"
#include "serverAgentEntitySpawn.hpp"

namespace packet::parsing {

ServerAgentEntitySpawn::ServerAgentEntitySpawn(const PacketContainer &packet,
                                   const sro::pk2::CharacterData &characterData,
                                   const sro::pk2::ItemData &itemData,
                                   const sro::pk2::SkillData &skillData,
                                   const sro::pk2::TeleportData &teleportData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  entity_ = parseSpawn(stream, characterData, itemData, skillData, teleportData);
  if (entity_) {
    // TODO: Handle "skill objects", like the recovery circle (will be nullptr)
    if (entity_->typeId1 == 1 || entity_->typeId1 == 4) {
      //BIONIC and STORE
      uint8_t spawnType = stream.Read<uint8_t>(); // 1=COS_SUMMON, 3=SPAWN, 4=SPAWN_WALK
    } else if (entity_->typeId1 == 3) {
      uint8_t dropSource = stream.Read<uint8_t>();
      uint32_t dropperUniqueId = stream.Read<uint32_t>();
    }
  }
}

std::shared_ptr<entity::Entity> ServerAgentEntitySpawn::entity() const {
  return entity_;
}

} // namespace packet::parsing