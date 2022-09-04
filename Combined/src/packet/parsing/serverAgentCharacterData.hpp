#ifndef PACKET_PARSING_SERVER_AGENT_CHARACTER_DATA_HPP
#define PACKET_PARSING_SERVER_AGENT_CHARACTER_DATA_HPP

#include "parsedPacket.hpp"
#include "../enums./packetEnums.hpp"
#include "../structures/packetInnerStructures.hpp"
#include "../../pk2/itemData.hpp"
#include "../../pk2/skillData.hpp"
#include "../../shared/silkroad_security.h"

#include <silkroad_lib/position.h>

#include <cstdint>
#include <map>
#include <memory>

namespace packet::parsing {
  
class ParsedServerAgentCharacterData : public ParsedPacket {
public:
  ParsedServerAgentCharacterData(const PacketContainer &packet, const pk2::ItemData &itemData, const pk2::SkillData &skillData);
  uint32_t refObjId() const;
  uint8_t curLevel() const;
  uint64_t currentExperience() const;
  uint32_t currentSpExperience() const;
  uint64_t gold() const;
  uint32_t skillPoints() const;
  uint32_t entityUniqueId() const;
  uint32_t hp() const;
  uint32_t mp() const;
  uint8_t inventorySize() const;
  const std::map<uint8_t, std::shared_ptr<storage::Item>>& inventoryItemMap() const;
  uint8_t avatarInventorySize() const;
  const std::map<uint8_t, std::shared_ptr<storage::Item>>& avatarInventoryItemMap() const;
  const std::vector<structures::Mastery>& masteries() const;
  const std::vector<structures::Skill>& skills() const;
  sro::Position position() const;
  float walkSpeed() const;
  float runSpeed() const;
  float hwanSpeed() const;
  std::string characterName() const;
  enums::LifeState lifeState() const;
  enums::MotionState motionState() const;
  enums::BodyState bodyState() const;
private:
  uint32_t refObjId_;
  uint8_t curLevel_;
  uint64_t currentExperience_;
  uint32_t currentSpExperience_;
  uint64_t gold_;
  uint32_t skillPoints_;
  uint32_t entityUniqueId_;
  uint32_t hp_;
  uint32_t mp_;
  uint8_t inventorySize_;
  std::map<uint8_t, std::shared_ptr<storage::Item>> inventoryItemMap_;
  uint8_t avatarInventorySize_;
  std::map<uint8_t, std::shared_ptr<storage::Item>> avatarInventoryItemMap_;
  std::vector<structures::Mastery> masteries_;
  std::vector<structures::Skill> skills_;
  sro::Position position_;
  enums::LifeState lifeState_;
  enums::MotionState motionState_;
  enums::BodyState bodyState_;
  float walkSpeed_;
  float runSpeed_;
  float hwanSpeed_;
  std::string characterName_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_CHARACTER_DATA_HPP