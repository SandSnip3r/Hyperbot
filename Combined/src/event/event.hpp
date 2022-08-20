#ifndef EVENT_EVENT_HPP_
#define EVENT_EVENT_HPP_

#include <cstdint>
#include <optional>
#include <string>

namespace event {

enum class EventCode {
  kSpawned,
  kHpPotionCooldownEnded,
  kMpPotionCooldownEnded,
  kVigorPotionCooldownEnded,
  kUniversalPillCooldownEnded,
  kPurificationPillCooldownEnded,
  kHpPercentChanged,
  kMpPercentChanged,
  kStatesChanged,
  kSkillCooldownEnded,
  kInventoryUpdated,
  kStorageUpdated,
  kSkillCastAboutToEnd,
  kKnockbackStatusEnded,
  kCharacterSpeedUpdated,
  kItemWaitForReuseDelay,
  kInjectPacket,
  kMovementTimerEnded,
  kStartTraining,
  kStopTraining,
  kEntityDeselected,
  kEntitySelected,
  kNpcTalkStart,
  kStorageOpened,
  kRepairSuccessful,
  kInventoryGoldUpdated,
  kStorageGoldUpdated,
  kGuildStorageGoldUpdated,
  kCharacterSkillPointsUpdated,
  kCharacterExperienceUpdated,

  // ===================================State updates===================================
  kStateUpdated = 0x1000,
  // Login state updates
  kStateShardIdUpdated,
  kStateConnectedToAgentServerUpdated,
  kStateReceivedCaptchaPromptUpdated,
  kStateCharacterListUpdated,
  // Movement state updates
  kMovementEnded,
  // ===================================================================================
};

struct Event {
public:
  explicit Event(EventCode code);
  const EventCode eventCode;
  virtual ~Event() = default;
};

struct SkillCooldownEnded : public Event {
public:
  SkillCooldownEnded(int32_t skillId);
  const int32_t skillRefId;
  virtual ~SkillCooldownEnded() = default;
};

struct InventoryUpdated : public Event {
public:
  InventoryUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~InventoryUpdated() = default;
};

struct StorageUpdated : public Event {
public:
  StorageUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~StorageUpdated() = default;
};

struct ItemWaitForReuseDelay : public Event {
public:
  ItemWaitForReuseDelay(uint8_t slotNum, uint16_t typeId);
  uint8_t inventorySlotNum;
  uint16_t itemTypeId;
  virtual ~ItemWaitForReuseDelay() = default;
};

struct InjectPacket : public Event {
public:
  enum class Direction { kClientToServer, kServerToClient };
  InjectPacket(Direction dir, uint16_t op, const std::string &d);
  Direction direction;
  uint16_t opcode;
  std::string data;
  virtual ~InjectPacket() = default;
};

} // namespace event

#endif // EVENT_EVENT_HPP_