#ifndef EVENT_EVENT_HPP_
#define EVENT_EVENT_HPP_

#include "packet/building/commonBuilding.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include "ui-proto/config.pb.h"

#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace event {

enum class EventCode {
  kLoggedIn,
  kSpawned,
  kCosSpawned,
  kItemCooldownEnded,
  kEntityHpChanged,
  kMpChanged,
  kMaxHpMpChanged,
  kStatsChanged,
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
  kRequestStartTraining,
  kRequestStopTraining,
  kTrainingStarted,
  kTrainingStopped,
  kEntityDeselected,
  kEntitySelected,
  kNpcTalkStart,
  kStorageInitialized,
  kGuildStorageInitialized,
  kRepairSuccessful,
  kInventoryGoldUpdated,
  kStorageGoldUpdated,
  kGuildStorageGoldUpdated,
  kCharacterLevelUpdated,
  kCharacterSkillPointsUpdated,
  kCharacterAvailableStatPointsUpdated,
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
  kDealtDamage,
  kKilledEntity,
  kOurSkillFailed,
  kPlayerCharacterBuffAdded,
  kPlayerCharacterBuffRemoved,
  kOurCommandError,
  kItemUseTimeout,
  kSkillCastTimeout,
  kEntityMovementBegan,
  kStateMachineActiveTooLong,

  // Only used to directly update movement state of entity
  kEntityMovementTimerEnded,

  // Only used to send position changed to UI
  kEntityPositionUpdated,

  kEntityNotMovingAngleChanged,
  kEntityBodyStateChanged,
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
  kWalkingPathUpdated,

  kNewConfigReceived,
  kConfigUpdated,

  kInventoryItemUpdated,
  kHwanPointsUpdated,

  // Alchemy
  kAlchemyCompleted,
  kAlchemyTimedOut,

  kGmCommandTimedOut,
  kChatReceived,
  kGameReset,
  kSetCurrentPositionAsTrainingCenter,
  kResurrectOption,
  kLeveledUpSkill,

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

struct EntityBodyStateChanged : public Event {
public:
  EntityBodyStateChanged(sro::scalar_types::EntityGlobalId id);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityBodyStateChanged() = default;
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

struct DealtDamage : public Event {
public:
  DealtDamage(sro::scalar_types::EntityGlobalId sourceId, sro::scalar_types::EntityGlobalId targetId, uint32_t damageAmount);
  const sro::scalar_types::EntityGlobalId sourceId;
  const sro::scalar_types::EntityGlobalId targetId;
  const uint32_t damageAmount;
  virtual ~DealtDamage() = default;
};

struct KilledEntity : public Event {
public:
  KilledEntity(sro::scalar_types::EntityGlobalId targetId);
  const sro::scalar_types::EntityGlobalId targetId;
  virtual ~KilledEntity() = default;
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

struct BuffRemoved : public Event {
public:
  BuffRemoved(sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceObjectId buffId);
  const sro::scalar_types::EntityGlobalId entityGlobalId;
  const sro::scalar_types::ReferenceObjectId buffRefId;
  virtual ~BuffRemoved() = default;
};

struct CommandError : public Event {
public:
  CommandError(const packet::structures::ActionCommand &cmd);
  const packet::structures::ActionCommand command;
  virtual ~CommandError() = default;
};
struct ItemUseTimeout : public Event {
public:
  ItemUseTimeout(uint8_t slot, type_id::TypeId tid);
  const uint8_t slotNum;
  const type_id::TypeId typeData;
  virtual ~ItemUseTimeout() = default;
};

struct SkillCastTimeout : public Event {
public:
  SkillCastTimeout(sro::scalar_types::ReferenceObjectId skillId);
  const sro::scalar_types::ReferenceObjectId skillId;
  virtual ~SkillCastTimeout() = default;
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

struct WalkingPathUpdated : public Event {
public:
  WalkingPathUpdated(const std::vector<packet::building::NetworkReadyPosition> &waypoints);
  const std::vector<packet::building::NetworkReadyPosition> waypoints;
  virtual ~WalkingPathUpdated() = default;
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

struct ChatReceived : public Event {
public:
  explicit ChatReceived(packet::enums::ChatType type, uint32_t senderGlobalId, const std::string &msg);
  explicit ChatReceived(packet::enums::ChatType type, const std::string &senderName, const std::string &msg);
  packet::enums::ChatType chatType;
  // Sender Global Id or Sender Name.
  std::variant<uint32_t, std::string> sender;
  std::string message;
  virtual ~ChatReceived() = default;
};

struct ConfigUpdated : public Event {
public:
  explicit ConfigUpdated(const proto::config::Config &c);
  proto::config::Config config;
  virtual ~ConfigUpdated() = default;
};

struct ResurrectOption : public Event {
public:
  explicit ResurrectOption(packet::enums::ResurrectionOptionFlag option);
  packet::enums::ResurrectionOptionFlag option;
  virtual ~ResurrectOption() = default;
};

struct LeveledUpSkill : public Event {
public:
  explicit LeveledUpSkill(sro::scalar_types::ReferenceSkillId oldSkillId, sro::scalar_types::ReferenceSkillId newSkillId);
  sro::scalar_types::ReferenceSkillId oldSkillRefId;
  sro::scalar_types::ReferenceSkillId newSkillRefId;
  virtual ~LeveledUpSkill() = default;
};

} // namespace event

#endif // EVENT_EVENT_HPP_