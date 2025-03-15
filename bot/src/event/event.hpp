#ifndef EVENT_EVENT_HPP_
#define EVENT_EVENT_HPP_

#include "clientManagerInterface.hpp"
#include "common/pvpDescriptor.hpp"
#include "common/sessionId.hpp"
#include "packet/building/commonBuilding.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#define EVENT_EVENTCODE_LIST(F) \
  F(SelfSpawned) \
  F(CosSpawned) \
  F(InternalItemCooldownEnded) \
  F(ItemCooldownEnded) \
  F(EntityHpChanged) \
  F(EntityMpChanged) \
  F(MaxHpMpChanged) \
  F(StatsChanged) \
  F(EntityStatesChanged) \
  F(InternalSkillCooldownEnded) \
  F(SkillCooldownEnded) \
  F(InventoryUpdated) \
  F(AvatarInventoryUpdated) \
  F(CosInventoryUpdated) \
  F(StorageUpdated) \
  F(GuildStorageUpdated) \
  F(SkillCastAboutToEnd) \
  F(ItemUseSuccess) \
  F(ItemUseFailed) \
  F(ItemUseTimeout) \
  F(InjectPacket) \
  F(RequestStartTraining) \
  F(RequestStopTraining) \
  F(TrainingStarted) \
  F(TrainingStopped) \
  F(EntityDeselected) \
  F(EntitySelected) \
  F(NpcTalkStart) \
  F(StorageInitialized) \
  F(GuildStorageInitialized) \
  F(RepairSuccessful) \
  F(InventoryGoldUpdated) \
  F(StorageGoldUpdated) \
  F(GuildStorageGoldUpdated) \
  F(CharacterLevelUpdated) \
  F(CharacterSkillPointsUpdated) \
  F(CharacterAvailableStatPointsUpdated) \
  F(CharacterExperienceUpdated) \
  F(EnteredNewRegion) \
  F(EntitySpawned) \
  F(EntityDespawned) \
  F(EntityMovementEnded) \
  F(EntityOwnershipRemoved) \
  F(StateMachineCreated) \
  F(StateMachineDestroyed) \
  F(SkillBegan) \
  F(SkillEnded) \
  F(DealtDamage) \
  F(KilledEntity) \
  F(SkillFailed) \
  F(PlayerCharacterBuffAdded) \
  F(PlayerCharacterBuffRemoved) \
  F(CommandError) \
  F(CommandSkipped) \
  F(SkillCastTimeout) \
  F(EntityMovementBegan) \
  F(StateMachineActiveTooLong) \
  F(EntityMovementTimerEnded) \
  F(EntityPositionUpdated) \
  F(EntityNotMovingAngleChanged) \
  F(EntityBodyStateChanged) \
  F(EntityLifeStateChanged) \
  F(EntityEnteredGeometry) \
  F(EntityExitedGeometry) \
  F(TrainingAreaSet) \
  F(TrainingAreaReset) \
  F(KnockedBack) \
  F(KnockedDown) \
  F(KnockbackStunEnded) \
  F(KnockdownStunEnded) \
  F(MovementRequestTimedOut) \
  F(WalkingPathUpdated) \
  F(InventoryItemUpdated) \
  F(HwanPointsUpdated) \
  F(AlchemyCompleted) \
  F(AlchemyTimedOut) \
  F(GmCommandTimedOut) \
  F(ChatReceived) \
  F(SetCurrentPositionAsTrainingCenter) \
  F(ResurrectOption) \
  F(LearnMasterySuccess) \
  F(LearnSkillSuccess) \
  F(LearnSkillError) \
  F(Timeout) \
  F(LoginCompleted) \
  F(RlStartPvp) \
  F(StateUpdated) \
  F(ServerAuthSuccess) \
  F(GatewayPatchResponseReceived) \
  F(ShardListReceived) \
  F(GatewayLoginResponseReceived) \
  F(ConnectedToAgentServer) \
  F(CharacterListReceived) \
  F(IbuvChallengeReceived) \
  F(CharacterSelectionJoinSuccess) \
  F(OperatorRequestSuccess) \
  F(OperatorRequestError) \
  F(EquipCountdownStart) \
  F(FreePvpUpdateSuccess) \
  F(PvpManagerReadyForAssignment) \
  F(BeginPvp) \
  F(ReadyForPvp) \
  F(DispelSuccess) \
  F(ClientDied) \
  F(StarRlTraining) \
  F(Dummy)

namespace event {

enum class EventCode {
#define F(name) k##name,
  EVENT_EVENTCODE_LIST(F)
#undef F
};

std::string_view toString(EventCode eventCode);

struct Event {
public:
  using EventId = uint32_t;
  explicit Event(EventId id, EventCode code);
  const EventId eventId;
  const EventCode eventCode;
  virtual ~Event() = default;
};

struct SessionSpecificEvent : public Event {
public:
  explicit SessionSpecificEvent(EventId id, EventCode code, SessionId sessionId);
  const SessionId sessionId;
  virtual ~SessionSpecificEvent() = default;
};

struct ServerAuthSuccess : public SessionSpecificEvent {
public:
  ServerAuthSuccess(EventId id, SessionId sessionId);
  virtual ~ServerAuthSuccess() = default;
};

struct GatewayPatchResponseReceived : public SessionSpecificEvent {
public:
  GatewayPatchResponseReceived(EventId id, SessionId sessionId);
  virtual ~GatewayPatchResponseReceived() = default;
};

struct ShardListReceived : public SessionSpecificEvent {
public:
  ShardListReceived(EventId id, SessionId sessionId, const std::vector<packet::structures::Shard> &shards);
  const std::vector<packet::structures::Shard> shards;
  virtual ~ShardListReceived() = default;
};

struct IbuvChallengeReceived : public SessionSpecificEvent {
public:
  IbuvChallengeReceived(EventId id, SessionId sessionId);
  virtual ~IbuvChallengeReceived() = default;
};

struct GatewayLoginResponseReceived : public SessionSpecificEvent {
public:
  GatewayLoginResponseReceived(EventId id, SessionId sessionId, uint32_t agentServerToken);
  const uint32_t agentServerToken;
  virtual ~GatewayLoginResponseReceived() = default;
};

struct ConnectedToAgentServer : public SessionSpecificEvent {
public:
  ConnectedToAgentServer(EventId id, SessionId sessionId);
  virtual ~ConnectedToAgentServer() = default;
};

struct CharacterListReceived : public SessionSpecificEvent {
public:
  CharacterListReceived(EventId id, SessionId sessionId, const std::vector<packet::structures::character_selection::Character> &characters);
  const std::vector<packet::structures::character_selection::Character> characters;
  virtual ~CharacterListReceived() = default;
};

struct CharacterSelectionJoinSuccess : public SessionSpecificEvent {
public:
  CharacterSelectionJoinSuccess(EventId id, SessionId sessionId);
  virtual ~CharacterSelectionJoinSuccess() = default;
};

struct SelfSpawned : public SessionSpecificEvent {
public:
  SelfSpawned(EventId id, SessionId sessionId, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~SelfSpawned() = default;
};

// This event is intended only for an entity to subscribe to.
struct InternalSkillCooldownEnded : public Event {
public:
  InternalSkillCooldownEnded(EventId id, sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceSkillId skillId);
  const sro::scalar_types::EntityGlobalId globalId;
  const sro::scalar_types::ReferenceSkillId skillRefId;
  virtual ~InternalSkillCooldownEnded() = default;
};

// This is the event that an entity publishes after it handles InternalSkillColldownEnded.
struct SkillCooldownEnded : public Event {
public:
  SkillCooldownEnded(EventId id, sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceSkillId skillId);
  const sro::scalar_types::EntityGlobalId globalId;
  const sro::scalar_types::ReferenceSkillId skillRefId;
  virtual ~SkillCooldownEnded() = default;
};

struct InventoryUpdated : public Event {
public:
  InventoryUpdated(EventId id, sro::scalar_types::EntityGlobalId globalId, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  const sro::scalar_types::EntityGlobalId globalId;
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~InventoryUpdated() = default;
};

struct AvatarInventoryUpdated : public Event {
public:
  AvatarInventoryUpdated(EventId id, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~AvatarInventoryUpdated() = default;
};

struct CosInventoryUpdated : public Event {
public:
  CosInventoryUpdated(EventId id, sro::scalar_types::EntityGlobalId gId, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  sro::scalar_types::EntityGlobalId globalId;
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~CosInventoryUpdated() = default;
};

struct StorageUpdated : public Event {
public:
  StorageUpdated(EventId id, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~StorageUpdated() = default;
};

struct GuildStorageUpdated : public Event {
public:
  GuildStorageUpdated(EventId id, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot);
  const std::optional<int8_t> srcSlotNum;
  const std::optional<int8_t> destSlotNum;
  virtual ~GuildStorageUpdated() = default;
};

struct ItemUseSuccess : public Event {
public:
  ItemUseSuccess(EventId id, sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::StorageIndexType slotNum, sro::scalar_types::ReferenceObjectId refId);
  sro::scalar_types::EntityGlobalId globalId;
  sro::scalar_types::StorageIndexType slotNum;
  sro::scalar_types::ReferenceObjectId refId;
  virtual ~ItemUseSuccess() = default;
};

struct ItemUseFailed : public Event {
public:
  ItemUseFailed(EventId id, uint8_t slotNum, type_id::TypeId typeId, packet::enums::InventoryErrorCode reason_param);
  uint8_t inventorySlotNum;
  type_id::TypeId itemTypeId;
  packet::enums::InventoryErrorCode reason;
  virtual ~ItemUseFailed() = default;
};

struct ItemUseTimeout : public Event {
public:
  ItemUseTimeout(EventId id, uint8_t slot, type_id::TypeId tid);
  const uint8_t slotNum;
  const type_id::TypeId typeData;
  virtual ~ItemUseTimeout() = default;
};

struct InjectPacket : public Event {
public:
  enum class Direction { kClientToServer, kServerToClient };
  InjectPacket(EventId id, Direction dir, uint16_t op, const std::string &d);
  Direction direction;
  uint16_t opcode;
  std::string data;
  virtual ~InjectPacket() = default;
};

struct CosSpawned : public Event {
public:
  CosSpawned(EventId id, sro::scalar_types::EntityGlobalId cosGId);
  const sro::scalar_types::EntityGlobalId cosGlobalId;
  virtual ~CosSpawned() = default;
};

struct EnteredNewRegion : public Event {
public:
  EnteredNewRegion(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EnteredNewRegion() = default;
};

struct EntitySpawned : public Event {
public:
  EntitySpawned(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntitySpawned() = default;
};

struct EntityDespawned : public Event {
public:
  EntityDespawned(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityDespawned() = default;
};

struct EntityMovementEnded : public Event {
public:
  EntityMovementEnded(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityMovementEnded() = default;
};

struct EntityMovementBegan : public Event {
public:
  EntityMovementBegan(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityMovementBegan() = default;
};

struct EntityMovementTimerEnded : public Event {
public:
  EntityMovementTimerEnded(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityMovementTimerEnded() = default;
};

struct EntityPositionUpdated : public Event {
public:
  EntityPositionUpdated(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityPositionUpdated() = default;
};

struct EntityNotMovingAngleChanged : public Event {
public:
  EntityNotMovingAngleChanged(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityNotMovingAngleChanged() = default;
};

struct EntityBodyStateChanged : public Event {
public:
  EntityBodyStateChanged(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityBodyStateChanged() = default;
};

struct EntityLifeStateChanged : public Event {
public:
  EntityLifeStateChanged(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityLifeStateChanged() = default;
};

struct EntityEnteredGeometry : public Event {
public:
  EntityEnteredGeometry(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityEnteredGeometry() = default;
};

struct EntityExitedGeometry : public Event {
public:
  EntityExitedGeometry(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityExitedGeometry() = default;
};

struct SkillBegan : public Event {
public:
  SkillBegan(EventId id, sro::scalar_types::EntityGlobalId casterId, sro::scalar_types::ReferenceObjectId skillId);
  const sro::scalar_types::EntityGlobalId casterGlobalId;
  const sro::scalar_types::ReferenceObjectId skillRefId;
  virtual ~SkillBegan() = default;
};

struct SkillEnded : public Event {
public:
  SkillEnded(EventId id, sro::scalar_types::EntityGlobalId casterId, sro::scalar_types::ReferenceObjectId skillId);
  const sro::scalar_types::EntityGlobalId casterGlobalId;
  const sro::scalar_types::ReferenceObjectId skillRefId;
  virtual ~SkillEnded() = default;
};

struct DealtDamage : public Event {
public:
  DealtDamage(EventId id, sro::scalar_types::EntityGlobalId sourceId, sro::scalar_types::EntityGlobalId targetId, uint32_t damageAmount);
  const sro::scalar_types::EntityGlobalId sourceId;
  const sro::scalar_types::EntityGlobalId targetId;
  const uint32_t damageAmount;
  virtual ~DealtDamage() = default;
};

struct KilledEntity : public Event {
public:
  KilledEntity(EventId id, sro::scalar_types::EntityGlobalId targetId);
  const sro::scalar_types::EntityGlobalId targetId;
  virtual ~KilledEntity() = default;
};

struct SkillFailed : public Event {
public:
  SkillFailed(EventId id, sro::scalar_types::EntityGlobalId casterGlobalId, sro::scalar_types::ReferenceSkillId skillId, uint16_t err);
  const sro::scalar_types::EntityGlobalId casterGlobalId;
  const sro::scalar_types::ReferenceSkillId skillRefId;
  const uint16_t errorCode;
  virtual ~SkillFailed() = default;
};

struct EntityHpChanged : public Event {
public:
  EntityHpChanged(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityHpChanged() = default;
};

struct EntityMpChanged : public Event {
public:
  EntityMpChanged(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityMpChanged() = default;
};

struct EntityStatesChanged : public Event {
public:
  EntityStatesChanged(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityStatesChanged() = default;
};

struct BuffAdded : public Event {
public:
  BuffAdded(EventId id, sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceSkillId buffId);
  const sro::scalar_types::EntityGlobalId entityGlobalId;
  const sro::scalar_types::ReferenceSkillId buffRefId;
  virtual ~BuffAdded() = default;
};

struct BuffRemoved : public Event {
public:
  BuffRemoved(EventId id, sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceSkillId buffId);
  const sro::scalar_types::EntityGlobalId entityGlobalId;
  const sro::scalar_types::ReferenceSkillId buffRefId;
  virtual ~BuffRemoved() = default;
};

struct CommandError : public Event {
public:
  CommandError(EventId id, sro::scalar_types::EntityGlobalId issuingGlobalId, const packet::structures::ActionCommand &cmd);
  const sro::scalar_types::EntityGlobalId issuingGlobalId;
  const packet::structures::ActionCommand command;
  virtual ~CommandError() = default;
};

struct CommandSkipped : public Event {
public:
  CommandSkipped(EventId id, sro::scalar_types::EntityGlobalId issuingGlobalId, const packet::structures::ActionCommand &cmd);
  const sro::scalar_types::EntityGlobalId issuingGlobalId;
  const packet::structures::ActionCommand command;
  virtual ~CommandSkipped() = default;
};

struct SkillCastTimeout : public Event {
public:
  SkillCastTimeout(EventId id, sro::scalar_types::ReferenceObjectId skillId);
  const sro::scalar_types::ReferenceObjectId skillId;
  virtual ~SkillCastTimeout() = default;
};

struct EntityOwnershipRemoved : public Event {
public:
  EntityOwnershipRemoved(EventId id, sro::scalar_types::EntityGlobalId globalId);
  const sro::scalar_types::EntityGlobalId globalId;
  virtual ~EntityOwnershipRemoved() = default;
};

struct StateMachineCreated : public Event {
public:
  StateMachineCreated(EventId id, const std::string &name);
  const std::string stateMachineName;
  virtual ~StateMachineCreated() = default;
};

struct InternalItemCooldownEnded : public Event {
public:
  InternalItemCooldownEnded(EventId eventId, sro::scalar_types::EntityGlobalId globalId, type_id::TypeId typeId);
  const sro::scalar_types::EntityGlobalId globalId;
  const type_id::TypeId typeId;
  virtual ~InternalItemCooldownEnded() = default;
};

struct ItemCooldownEnded : public Event {
public:
  ItemCooldownEnded(EventId eventId, sro::scalar_types::EntityGlobalId globalId, type_id::TypeId typeId);
  const sro::scalar_types::EntityGlobalId globalId;
  const type_id::TypeId typeId;
  virtual ~ItemCooldownEnded() = default;
};

struct WalkingPathUpdated : public Event {
public:
  WalkingPathUpdated(EventId id, const std::vector<packet::building::NetworkReadyPosition> &waypoints);
  const std::vector<packet::building::NetworkReadyPosition> waypoints;
  virtual ~WalkingPathUpdated() = default;
};

struct InventoryItemUpdated : public Event {
public:
  InventoryItemUpdated(EventId id, const uint8_t &slot);
  const uint8_t slotIndex;
  virtual ~InventoryItemUpdated() = default;
};

struct ChatReceived : public Event {
public:
  explicit ChatReceived(EventId id, packet::enums::ChatType type, sro::scalar_types::EntityGlobalId senderGlobalId, const std::string &msg);
  explicit ChatReceived(EventId id, packet::enums::ChatType type, const std::string &senderName, const std::string &msg);
  packet::enums::ChatType chatType;
  // Sender Global Id or Sender Name.
  std::variant<sro::scalar_types::EntityGlobalId, std::string> sender;
  std::string message;
  virtual ~ChatReceived() = default;
};

struct ResurrectOption : public Event {
public:
  explicit ResurrectOption(EventId id, packet::enums::ResurrectionOptionFlag option);
  packet::enums::ResurrectionOptionFlag option;
  virtual ~ResurrectOption() = default;
};

struct LearnMasterySuccess : public Event {
public:
  explicit LearnMasterySuccess(EventId id, sro::scalar_types::ReferenceMasteryId masteryId);
  sro::scalar_types::ReferenceMasteryId masteryId;
  virtual ~LearnMasterySuccess() = default;
};

struct LearnSkillSuccess : public Event {
public:
  explicit LearnSkillSuccess(EventId id, sro::scalar_types::ReferenceSkillId newSkillId, std::optional<sro::scalar_types::ReferenceSkillId> oldSkillId = std::nullopt);
  sro::scalar_types::ReferenceSkillId newSkillRefId;
  std::optional<sro::scalar_types::ReferenceSkillId> oldSkillRefId;
  virtual ~LearnSkillSuccess() = default;
};

struct OperatorRequestSuccess : public Event {
  explicit OperatorRequestSuccess(EventId id, sro::scalar_types::EntityGlobalId globalId, packet::enums::OperatorCommand operatorCommand);
  sro::scalar_types::EntityGlobalId globalId;
  packet::enums::OperatorCommand operatorCommand;
  virtual ~OperatorRequestSuccess() = default;
};

struct OperatorRequestError : public Event {
  explicit OperatorRequestError(EventId id, sro::scalar_types::EntityGlobalId globalId, packet::enums::OperatorCommand operatorCommand);
  sro::scalar_types::EntityGlobalId globalId;
  packet::enums::OperatorCommand operatorCommand;
  virtual ~OperatorRequestError() = default;
};

struct EquipCountdownStart : public Event {
  explicit EquipCountdownStart(EventId id, sro::scalar_types::EntityGlobalId globalId);
  sro::scalar_types::EntityGlobalId globalId;
  virtual ~EquipCountdownStart() = default;
};

struct FreePvpUpdateSuccess : public Event {
  explicit FreePvpUpdateSuccess(EventId id, sro::scalar_types::EntityGlobalId globalId);
  sro::scalar_types::EntityGlobalId globalId;
  virtual ~FreePvpUpdateSuccess() = default;
};

struct PvpManagerReadyForAssignment : public Event {
  explicit PvpManagerReadyForAssignment(EventId id, SessionId sessionId);
  SessionId sessionId;
  virtual ~PvpManagerReadyForAssignment() = default;
};

struct BeginPvp : public Event {
  explicit BeginPvp(EventId id, common::PvpDescriptor pvpDescriptor);
  common::PvpDescriptor pvpDescriptor;
  virtual ~BeginPvp() = default;
};

struct ReadyForPvp : public Event {
  explicit ReadyForPvp(EventId id, sro::scalar_types::EntityGlobalId globalId);
  sro::scalar_types::EntityGlobalId globalId;
  virtual ~ReadyForPvp() = default;
};

struct ClientDied : public Event {
  explicit ClientDied(EventId id, ClientManagerInterface::ClientId clientId);
  ClientManagerInterface::ClientId clientId;
  virtual ~ClientDied() = default;
};

} // namespace event

#endif // EVENT_EVENT_HPP_