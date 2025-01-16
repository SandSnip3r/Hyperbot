#include "packet/parsing/commonParsing.hpp"
#include "serverAgentEntityGroupSpawnData.hpp"

namespace packet::parsing {

ServerAgentEntityGroupSpawnData::ServerAgentEntityGroupSpawnData(const PacketContainer &packet,
                                                                 const pk2::CharacterData &characterData,
                                                                 const pk2::ItemData &itemData,
                                                                 const pk2::SkillData &skillData,
                                                                 const pk2::TeleportData &teleportData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  // This data is originally from the begin packet (before the data packet)
  groupSpawnType_ = static_cast<enums::GroupSpawnType>(stream.Read<uint8_t>());
  uint16_t groupSpawnAmount = stream.Read<uint16_t>();
  if (groupSpawnType_ == enums::GroupSpawnType::kSpawn) {
    for (int spawnNum=0; spawnNum<groupSpawnAmount; ++spawnNum) {
      auto entity = parseSpawn(stream, characterData, itemData, skillData, teleportData);
      entities_.emplace_back(entity);
    }
  } else if (groupSpawnType_ == enums::GroupSpawnType::kDespawn) {
    for (int despawnNum=0; despawnNum<groupSpawnAmount; ++despawnNum) {
      despawnGlobalIds_.emplace_back(stream.Read<sro::scalar_types::EntityGlobalId>());
    }
  }
}

enums::GroupSpawnType ServerAgentEntityGroupSpawnData::groupSpawnType() const {
  return groupSpawnType_;
}

const std::vector<std::shared_ptr<entity::Entity>>& ServerAgentEntityGroupSpawnData::entities() const {
  return entities_;
}

const std::vector<sro::scalar_types::EntityGlobalId>& ServerAgentEntityGroupSpawnData::despawnGlobalIds() const {
  return despawnGlobalIds_;
}

} // namespace packet::parsing