#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_GROUP_SPAWN_DATA_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_GROUP_SPAWN_DATA_HPP

#include "entity/entity.hpp"
#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/parsedPacket.hpp"

#include <memory>
#include <vector>

namespace packet::parsing {

class ServerAgentEntityGroupSpawnData : public ParsedPacket {
public:
  ServerAgentEntityGroupSpawnData(const PacketContainer &packet,
                                  const pk2::CharacterData &characterData,
                                  const pk2::ItemData &itemData,
                                  const pk2::SkillData &skillData,
                                  const pk2::TeleportData &teleportData);
  enums::GroupSpawnType groupSpawnType() const;
  const std::vector<std::shared_ptr<entity::Entity>>& entities() const;
  const std::vector<sro::scalar_types::EntityGlobalId>& despawnGlobalIds() const;
private:
  enums::GroupSpawnType groupSpawnType_;
  std::vector<std::shared_ptr<entity::Entity>> entities_;
  std::vector<uint32_t> despawnGlobalIds_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_GROUP_SPAWN_DATA_HPP