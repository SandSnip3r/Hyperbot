#ifndef PACKET_PARSING_SERVER_AGENT_CHARACTER_DATA_HPP
#define PACKET_PARSING_SERVER_AGENT_CHARACTER_DATA_HPP

#include "entity/mobileEntity.hpp"
#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "pk2/itemData.hpp"
#include "pk2/skillData.hpp"
#include "shared/silkroad_security.h"

#include <silkroad_lib/entity.hpp>
#include <silkroad_lib/position.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace packet::parsing {

class ServerAgentCharacterData : public ParsedPacket {
public:
  ServerAgentCharacterData(const PacketContainer &packet, const pk2::ItemData &itemData, const pk2::SkillData &skillData);
  sro::scalar_types::ReferenceObjectId refObjId() const { return refObjId_; }
  uint8_t curLevel() const;
  uint64_t currentExperience() const;
  uint32_t currentSpExperience() const;
  uint64_t gold() const;
  uint32_t skillPoints() const;
  uint16_t availableStatPoints() const;
  uint8_t hwanPoints() const;
  sro::scalar_types::EntityGlobalId globalId() const { return globalId_; }
  uint32_t hp() const;
  uint32_t mp() const;
  uint8_t hwanLevel() const;
  uint8_t inventorySize() const;
  const std::map<uint8_t, std::shared_ptr<storage::Item>>& inventoryItemMap() const;
  uint8_t avatarInventorySize() const;
  const std::map<uint8_t, std::shared_ptr<storage::Item>>& avatarInventoryItemMap() const;
  const std::vector<structures::Mastery>& masteries() const;
  const std::vector<structures::Skill>& skills() const;
  sro::Position position() const;
  sro::Angle angle() const;
  float walkSpeed() const;
  float runSpeed() const;
  float hwanSpeed() const;
  std::string characterName() const;
  sro::entity::LifeState lifeState() const;
  entity::MotionState motionState() const;
  enums::BodyState bodyState() const;
  uint32_t jId() const;
private:
  sro::scalar_types::ReferenceObjectId refObjId_;
  uint8_t curLevel_;
  uint64_t currentExperience_;
  uint32_t currentSpExperience_;
  uint64_t gold_;
  uint32_t skillPoints_;
  uint16_t availableStatPoints_;
  uint8_t hwanPoints_;
  sro::scalar_types::EntityGlobalId globalId_;
  uint32_t hp_;
  uint32_t mp_;
  uint8_t hwanLevel_;
  uint8_t inventorySize_;
  std::map<uint8_t, std::shared_ptr<storage::Item>> inventoryItemMap_;
  uint8_t avatarInventorySize_;
  std::map<uint8_t, std::shared_ptr<storage::Item>> avatarInventoryItemMap_;
  std::vector<structures::Mastery> masteries_;
  std::vector<structures::Skill> skills_;
  sro::Position position_;
  sro::Angle angle_;
  sro::entity::LifeState lifeState_;
  entity::MotionState motionState_;
  enums::BodyState bodyState_;
  float walkSpeed_;
  float runSpeed_;
  float hwanSpeed_;
  std::string characterName_;
  uint32_t jId_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_CHARACTER_DATA_HPP