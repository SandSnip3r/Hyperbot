#ifndef EVENT_EVENT_HPP_
#define EVENT_EVENT_HPP_

#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <optional>
#include <string>

namespace event {

enum class EventCode {
  kSpawned,
  kCosSpawned,
  kHpPotionCooldownEnded,
  kMpPotionCooldownEnded,
  kVigorPotionCooldownEnded,
  kUniversalPillCooldownEnded,
  kPurificationPillCooldownEnded,
  kHpChanged,
  kMpChanged,
  kMaxHpMpChanged,
  kStatesChanged,
  kSkillCooldownEnded,
  kInventoryUpdated,
  kAvatarInventoryUpdated,
  kCosInventoryUpdated,
  kStorageUpdated,
  kGuildStorageUpdated,
  kSkillCastAboutToEnd,
  kKnockbackStatusEnded,
  kItemWaitForReuseDelay,
  kInjectPacket,
  kMovementTimerEnded,
  kStartTraining,
  kStopTraining,
  kEntityDeselected,
  kEntitySelected,
  kNpcTalkStart,
  kStorageInitialized,
  kGuildStorageInitialized,
  kRepairSuccessful,
  kInventoryGoldUpdated,
  kStorageGoldUpdated,
  kGuildStorageGoldUpdated,
  kCharacterSkillPointsUpdated,
  kCharacterExperienceUpdated,
  kEnteredNewRegion,
  kEntitySpawned,
  kEntityDespawned,
  kEntityMovementEnded,
  kEntityMovementBegan,
  kEntityMovementTimerEnded,
  kEntitySyncedPosition,

  // ===================================State updates===================================
  kStateUpdated = 0x1000,
  // Login state updates
  kStateShardIdUpdated,
  kStateConnectedToAgentServerUpdated,
  kStateReceivedCaptchaPromptUpdated,
  kStateCharacterListUpdated,
  // Movement state updates
  kMovementBegan,
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

struct AvatarInventoryUpdated : public Event {
public:
  AvatarInventoryUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~AvatarInventoryUpdated() = default;
};

struct CosInventoryUpdated : public Event {
public:
  CosInventoryUpdated(uint32_t gId, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  uint32_t globalId;
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~CosInventoryUpdated() = default;
};

struct StorageUpdated : public Event {
public:
  StorageUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~StorageUpdated() = default;
};

struct GuildStorageUpdated : public Event {
public:
  GuildStorageUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~GuildStorageUpdated() = default;
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

struct CosSpawned : public Event {
public:
  CosSpawned(uint32_t cosGId);
  const uint32_t cosGlobalId;
  virtual ~CosSpawned() = default;
};

struct EntitySpawned : public Event {
public:
  EntitySpawned(uint32_t id);
  const uint32_t globalId;
  virtual ~EntitySpawned() = default;
};

struct EntityDespawned : public Event {
public:
  EntityDespawned(uint32_t id);
  const uint32_t globalId;
  virtual ~EntityDespawned() = default;
};

struct EntityMovementEnded : public Event {
public:
  EntityMovementEnded(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityMovementEnded() = default;
};

struct EntityMovementBegan : public Event {
public:
  EntityMovementBegan(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityMovementBegan() = default;
};

struct EntityMovementTimerEnded : public Event {
public:
  EntityMovementTimerEnded(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityMovementTimerEnded() = default;
};

struct EntitySyncedPosition : public Event {
public:
  EntitySyncedPosition(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntitySyncedPosition() = default;
};

} // namespace event

#endif // EVENT_EVENT_HPP_