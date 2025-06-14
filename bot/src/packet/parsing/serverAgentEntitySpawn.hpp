#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_SPAWN_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_SPAWN_HPP

#include "entity/entity.hpp"
#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/parsedPacket.hpp"

#include <memory>

namespace packet::parsing {

class ServerAgentEntitySpawn : public ParsedPacket {
public:
public:
  ServerAgentEntitySpawn(const PacketContainer &packet,
                         const sro::pk2::CharacterData &characterData,
                         const sro::pk2::ItemData &itemData,
                         const sro::pk2::SkillData &skillData,
                         const sro::pk2::TeleportData &teleportData);
  std::shared_ptr<entity::Entity> entity() const;
private:
  std::shared_ptr<entity::Entity> entity_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_SPAWN_HPP