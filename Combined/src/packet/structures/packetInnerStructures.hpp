#ifndef PACKET_INNER_STRUCTURES_HPP
#define PACKET_INNER_STRUCTURES_HPP

#include "packet/enums/packetEnums.hpp"
#include "storage/item.hpp"

#include <cmath>
#include <ostream>
#include <string>
#include <vector>

namespace packet::structures {

namespace CharacterSelection {

struct Item {
  uint32_t refId;
  uint8_t plus;
};

struct Avatar {
  uint32_t refId;
  uint8_t plus;
};

struct Character {
public:
  uint32_t refObjID;
  // uint16_t  name.Length;
  std::string name;
  uint8_t scale;
  uint8_t curLevel;
  uint64_t expOffset;
  uint16_t strength;
  uint16_t intelligence;
  uint16_t statPoint;
  uint32_t curHP;
  uint32_t curMP;
  bool isDeleting;
    uint32_t charDeleteTime;
  uint8_t guildMemberClass;
  bool isGuildRenameRequired;
    // uint16_t currentGuildName.Length
    std::string currentGuildName;
  uint8_t academyMemberClass;
  // uint8_t itemCount;
  std::vector<Item> items;
  // uint8_t avatarItemCount;
  std::vector<Avatar> avatars;
};

} // namespace CharacterSelection

namespace vitals {

struct AbnormalState {
  uint32_t totalTime;
  uint16_t timeElapsed;
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
  Skill(uint32_t i, bool e) : id(i), enabled(e) {}
  uint32_t id;
  bool enabled;
};

struct SkillActionHitResult {
  enums::HitResult hitResult;
  enums::DamageFlag damageFlag;
  uint32_t damage;
  uint32_t effect;
  uint16_t regionId;
  float x, y, z;
};

struct SkillActionHitObject {
  uint32_t objGlobalId;
  std::vector<SkillActionHitResult> hits;
};

struct SkillAction {
  uint8_t actionFlag;
  std::vector<SkillActionHitObject> hitObjects;
  // If teleport or sprint
  uint16_t regionId;
  float x, y, z;
};

struct ActionCommand {
public:
  enums::CommandType commandType;
  enums::ActionType actionType;
  uint32_t refSkillId;
  enums::TargetType targetType;
  uint32_t targetGlobalId;
  uint16_t regionId;
  float x;
  float y;
  float z;
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

struct Position {
public:
  uint16_t regionId;
  float xOffset, yOffset, zOffset;
  bool isDungeon() const { return regionId & 0x8000; }
  uint8_t dungeonId() const { return regionId & 0xFF; }
  uint8_t xSector() const { return regionId & 0xFF; }
  uint8_t zSector() const { return (regionId >> 8) & 0x7F; }
};

} // namespace packet::structures

#endif // PACKET_INNER_STRUCTURES_HPP