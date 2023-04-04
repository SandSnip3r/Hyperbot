#ifndef EVENT_EVENT_HPP_
#define EVENT_EVENT_HPP_

#include "packet/structures/packetInnerStructures.hpp"

#include "ui-proto/config.pb.h"

#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <optional>
#include <string>

namespace event {

enum class EventCode {
  kLoggedIn,
  kSpawned,
  kCosSpawned,
  kItemCooldownEnded,
  kEntityHpChanged,
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
  kItemUseFailed,
  kInjectPacket,
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
  kEntityOwnershipRemoved,
  kStateMachineCreated,
  kStateMachineDestroyed,

  kSkillBegan,
  kSkillEnded,
  kOurSkillFailed,
  kOurBuffAdded,
  kOurBuffRemoved,
  kOurCommandError,
  // TODO: Refactor this whole itemUsedTimeout concept
  kItemUseTimeout,

  // Only used for sending to UI
  kEntityMovementBegan,

  // Only used to directly update movement state of entity
  kEntityMovementTimerEnded,

  // Only used to send position changed to UI
  kEntityPositionUpdated,

  kEntityNotMovingAngleChanged,
  kEntityLifeStateChanged,
  kEntityEnteredGeometry,
  kEntityExitedGeometry,
  kTrainingAreaSet,
  kTrainingAreaReset,

  kKnockedBack,
  kKnockedDown,
  kKnockbackStunEnded,
  kKnockdownStunEnded,

  kMovementRequestTimedOut,

  kNewConfigReceived,
  kConfigUpdated,

  kInventoryItemUpdated,

  // ===================================State updates===================================
  kStateUpdated = 0x1000,
  // Login state updates
  kStateShardIdUpdated,
  kStateConnectedToAgentServerUpdated,
  kStateReceivedCaptchaPromptUpdated,
  kStateCharacterListUpdated,
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

struct ItemUseFailed : public Event {
public:
  ItemUseFailed(uint8_t slotNum, type_id::TypeId typeId, packet::enums::InventoryErrorCode reason_param);
  uint8_t inventorySlotNum;
  type_id::TypeId itemTypeId;
  packet::enums::InventoryErrorCode reason;
  virtual ~ItemUseFailed() = default;
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

struct EntityPositionUpdated : public Event {
public:
  EntityPositionUpdated(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityPositionUpdated() = default;
};

struct EntityNotMovingAngleChanged : public Event {
public:
  EntityNotMovingAngleChanged(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityNotMovingAngleChanged() = default;
};

struct EntityLifeStateChanged : public Event {
public:
  EntityLifeStateChanged(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityLifeStateChanged() = default;
};

struct EntityEnteredGeometry : public Event {
public:
  EntityEnteredGeometry(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityEnteredGeometry() = default;
};

struct EntityExitedGeometry : public Event {
public:
  EntityExitedGeometry(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityExitedGeometry() = default;
};

struct SkillBegan : public Event {
public:
  SkillBegan(sro::scalar_types::EntityGlobalId casterId, sro::scalar_types::ReferenceObjectId skillId);
  const sro::scalar_types::EntityGlobalId casterGlobalId;
  const sro::scalar_types::ReferenceObjectId skillRefId;
  virtual ~SkillBegan() = default;
};

struct SkillEnded : public Event {
public:
  SkillEnded(sro::scalar_types::EntityGlobalId casterId, sro::scalar_types::ReferenceObjectId skillId);
  const sro::scalar_types::EntityGlobalId casterGlobalId;
  const sro::scalar_types::ReferenceObjectId skillRefId;
  virtual ~SkillEnded() = default;
};

struct OurSkillFailed : public Event {
public:
  OurSkillFailed(sro::scalar_types::ReferenceObjectId id, uint16_t err);
  const sro::scalar_types::ReferenceObjectId skillRefId;
  const uint16_t errorCode;
  virtual ~OurSkillFailed() = default;
};

struct EntityHpChanged : public Event {
public:
  EntityHpChanged(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityHpChanged() = default;
};

struct BuffAdded : public Event {
public:
  BuffAdded(sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceObjectId buffId);
  const sro::scalar_types::EntityGlobalId entityGlobalId;
  const sro::scalar_types::ReferenceObjectId buffRefId;
  virtual ~BuffAdded() = default;
};

struct CommandError : public Event {
public:
  CommandError(const packet::structures::ActionCommand &cmd);
  const packet::structures::ActionCommand command;
  virtual ~CommandError() = default;
};

// TODO: Refactor this whole itemUsedTimeout concept
struct ItemUseTimeout : public Event {
public:
  ItemUseTimeout(uint8_t slot, type_id::TypeId tid);
  const uint8_t slotNum;
  const type_id::TypeId typeData;
  virtual ~ItemUseTimeout() = default;
};

struct EntityOwnershipRemoved : public Event {
public:
  EntityOwnershipRemoved(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityOwnershipRemoved() = default;
};

struct StateMachineCreated : public Event {
public:
  StateMachineCreated(const std::string &name);
  const std::string stateMachineName;
  virtual ~StateMachineCreated() = default;
};

struct ItemCooldownEnded : public Event {
public:
  ItemCooldownEnded(type_id::TypeId tId);
  const type_id::TypeId typeId;
  virtual ~ItemCooldownEnded() = default;
};

struct NewConfigReceived : public Event {
public:
  NewConfigReceived(const proto::config::Config &config_param);
  const proto::config::Config config;
  virtual ~NewConfigReceived() = default;
};

struct InventoryItemUpdated : public Event {
public:
  InventoryItemUpdated(const uint8_t &slot);
  const uint8_t slotIndex;
  virtual ~InventoryItemUpdated() = default;
};

} // namespace event

#endif // EVENT_EVENT_HPP_