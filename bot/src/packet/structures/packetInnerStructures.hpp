#ifndef PACKET_INNER_STRUCTURES_HPP
#define PACKET_INNER_STRUCTURES_HPP

#include "packet/enums/packetEnums.hpp"
#include "storage/item.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <absl/strings/str_format.h>

#include <cmath>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace packet::structures {

struct Shard {
  uint16_t shardId;
  std::string shardName;
  uint16_t onlineCount;
  uint16_t capacity;
  uint8_t isOperating;
  uint8_t farmId;
  std::string toString() const {
    return absl::StrFormat("Shard %d: %s, %d/%d, %s, Farm %d",
                           shardId, shardName, onlineCount, capacity,
                           isOperating ? "Operating" : "Not Operating", farmId);
  }
};

namespace character_selection {

struct Item {
  sro::scalar_types::ReferenceObjectId refId;
  sro::scalar_types::OptLevelType plus;
};

struct Avatar {
  sro::scalar_types::ReferenceObjectId refId;
  sro::scalar_types::OptLevelType plus;
};

struct Character {
public:
  sro::scalar_types::ReferenceObjectId refObjID;
  std::string name;
  uint8_t scale;
  uint8_t curLevel;
  uint64_t expOffset;
  uint16_t strength;
  uint16_t intelligence;
  uint16_t statPoint;
  uint32_t curHP;
  uint32_t curMP;
  uint8_t isDeleting;
  uint32_t charDeleteRemainingMinutes; // if (isDeleting)
  uint8_t guildMemberClass;
  uint8_t isGuildRenameRequired;
  std::string currentGuildName; // if (isGuildRenameRequired)
  uint8_t academyMemberClass;
  std::vector<Item> items;
  std::vector<Avatar> avatarItems;
};

} // namespace character_selection

namespace vitals {

struct AbnormalState {
  uint32_t totalTime; // x100 = ms
  uint16_t timeElapsed; // x100 = ms
  uint16_t effectOrLevel;
};

} // namespace vitals

struct RentInfo {
  uint32_t rentType; // TODO: Enum for this
  uint16_t canDelete;
  uint32_t periodBeginTime;
  uint32_t periodEndTime;
  uint16_t canRecharge;
  uint32_t meterRateTime;
  uint32_t packingTime;
};

struct Mastery {
  Mastery(uint32_t i, uint8_t l) : id(i), level(l) {}
  uint32_t id;
  uint8_t level;
};

struct Skill {
  Skill(sro::scalar_types::ReferenceSkillId i, bool e) : id(i), enabled(e) {}
  sro::scalar_types::ReferenceSkillId id;
  bool enabled;
};

struct SkillActionHitResult {
  enums::HitResult hitResultFlag;
  enums::DamageFlag damageFlag;
  uint32_t damage;
  uint32_t effect;
  sro::Position position;
};

struct SkillActionHitObject {
  sro::scalar_types::EntityGlobalId targetGlobalId;
  std::vector<SkillActionHitResult> hits;
};

struct SkillAction {
  enums::ActionFlag actionFlag;
  std::vector<SkillActionHitObject> hitObjects;
  // If teleport or sprint
  sro::Position position;
};

struct ActionCommand {
public:
  enums::CommandType commandType;
  enums::ActionType actionType;
  sro::scalar_types::ReferenceObjectId refSkillId;
  enums::TargetType targetType;
  sro::scalar_types::EntityGlobalId targetGlobalId;
  sro::Position position;
  std::string toString() const;
};

struct ItemMovement {
  // TODO: Maybe move into Storage or something
  static constexpr uint8_t kGoldSlot = 0xFE;
  packet::enums::ItemMovementType type;
  uint8_t srcSlot, destSlot;
  uint16_t quantity;
  uint32_t goldPickAmount;
  uint64_t goldAmount;
  uint32_t globalId;
  uint8_t storeTabNumber;
  uint8_t storeSlotNumber;
  uint8_t stackCount;
  std::vector<uint8_t> destSlots;
  std::vector<structures::RentInfo> rentInfos;
  uint8_t buybackStackSize;
  std::shared_ptr<storage::Item> newItem;
};

} // namespace packet::structures

#endif // PACKET_INNER_STRUCTURES_HPP