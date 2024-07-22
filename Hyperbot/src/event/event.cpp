#include "event.hpp"

#include <absl/log/log.h>

namespace event {

std::string_view toString(EventCode eventCode) {
  if (eventCode == EventCode::kLoggedIn) {
    return "LoggedIn";
  }
  if (eventCode == EventCode::kSpawned) {
    return "Spawned";
  }
  if (eventCode == EventCode::kCosSpawned) {
    return "CosSpawned";
  }
  if (eventCode == EventCode::kItemCooldownEnded) {
    return "ItemCooldownEnded";
  }
  if (eventCode == EventCode::kEntityHpChanged) {
    return "EntityHpChanged";
  }
  if (eventCode == EventCode::kMpChanged) {
    return "MpChanged";
  }
  if (eventCode == EventCode::kMaxHpMpChanged) {
    return "MaxHpMpChanged";
  }
  if (eventCode == EventCode::kStatsChanged) {
    return "StatsChanged";
  }
  if (eventCode == EventCode::kStatesChanged) {
    return "StatesChanged";
  }
  if (eventCode == EventCode::kSkillCooldownEnded) {
    return "SkillCooldownEnded";
  }
  if (eventCode == EventCode::kInventoryUpdated) {
    return "InventoryUpdated";
  }
  if (eventCode == EventCode::kAvatarInventoryUpdated) {
    return "AvatarInventoryUpdated";
  }
  if (eventCode == EventCode::kCosInventoryUpdated) {
    return "CosInventoryUpdated";
  }
  if (eventCode == EventCode::kStorageUpdated) {
    return "StorageUpdated";
  }
  if (eventCode == EventCode::kGuildStorageUpdated) {
    return "GuildStorageUpdated";
  }
  if (eventCode == EventCode::kSkillCastAboutToEnd) {
    return "SkillCastAboutToEnd";
  }
  if (eventCode == EventCode::kItemUseFailed) {
    return "ItemUseFailed";
  }
  if (eventCode == EventCode::kInjectPacket) {
    return "InjectPacket";
  }
  if (eventCode == EventCode::kRequestStartTraining) {
    return "RequestStartTraining";
  }
  if (eventCode == EventCode::kRequestStopTraining) {
    return "RequestStopTraining";
  }
  if (eventCode == EventCode::kTrainingStarted) {
    return "TrainingStarted";
  }
  if (eventCode == EventCode::kTrainingStopped) {
    return "TrainingStopped";
  }
  if (eventCode == EventCode::kEntityDeselected) {
    return "EntityDeselected";
  }
  if (eventCode == EventCode::kEntitySelected) {
    return "EntitySelected";
  }
  if (eventCode == EventCode::kNpcTalkStart) {
    return "NpcTalkStart";
  }
  if (eventCode == EventCode::kStorageInitialized) {
    return "StorageInitialized";
  }
  if (eventCode == EventCode::kGuildStorageInitialized) {
    return "GuildStorageInitialized";
  }
  if (eventCode == EventCode::kRepairSuccessful) {
    return "RepairSuccessful";
  }
  if (eventCode == EventCode::kInventoryGoldUpdated) {
    return "InventoryGoldUpdated";
  }
  if (eventCode == EventCode::kStorageGoldUpdated) {
    return "StorageGoldUpdated";
  }
  if (eventCode == EventCode::kGuildStorageGoldUpdated) {
    return "GuildStorageGoldUpdated";
  }
  if (eventCode == EventCode::kCharacterLevelUpdated) {
    return "CharacterLevelUpdated";
  }
  if (eventCode == EventCode::kCharacterSkillPointsUpdated) {
    return "CharacterSkillPointsUpdated";
  }
  if (eventCode == EventCode::kCharacterAvailableStatPointsUpdated) {
    return "CharacterAvailableStatPointsUpdated";
  }
  if (eventCode == EventCode::kCharacterExperienceUpdated) {
    return "CharacterExperienceUpdated";
  }
  if (eventCode == EventCode::kEnteredNewRegion) {
    return "EnteredNewRegion";
  }
  if (eventCode == EventCode::kEntitySpawned) {
    return "EntitySpawned";
  }
  if (eventCode == EventCode::kEntityDespawned) {
    return "EntityDespawned";
  }
  if (eventCode == EventCode::kEntityMovementEnded) {
    return "EntityMovementEnded";
  }
  if (eventCode == EventCode::kEntityOwnershipRemoved) {
    return "EntityOwnershipRemoved";
  }
  if (eventCode == EventCode::kStateMachineCreated) {
    return "StateMachineCreated";
  }
  if (eventCode == EventCode::kStateMachineDestroyed) {
    return "StateMachineDestroyed";
  }
  if (eventCode == EventCode::kSkillBegan) {
    return "SkillBegan";
  }
  if (eventCode == EventCode::kSkillEnded) {
    return "SkillEnded";
  }
  if (eventCode == EventCode::kDealtDamage) {
    return "DealtDamage";
  }
  if (eventCode == EventCode::kKilledEntity) {
    return "KilledEntity";
  }
  if (eventCode == EventCode::kOurSkillFailed) {
    return "OurSkillFailed";
  }
  if (eventCode == EventCode::kPlayerCharacterBuffAdded) {
    return "PlayerCharacterBuffAdded";
  }
  if (eventCode == EventCode::kPlayerCharacterBuffRemoved) {
    return "PlayerCharacterBuffRemoved";
  }
  if (eventCode == EventCode::kOurCommandError) {
    return "OurCommandError";
  }
  if (eventCode == EventCode::kItemUseTimeout) {
    return "ItemUseTimeout";
  }
  if (eventCode == EventCode::kSkillCastTimeout) {
    return "SkillCastTimeout";
  }
  if (eventCode == EventCode::kEntityMovementBegan) {
    return "EntityMovementBegan";
  }
  if (eventCode == EventCode::kStateMachineActiveTooLong) {
    return "StateMachineActiveTooLong";
  }
  if (eventCode == EventCode::kEntityMovementTimerEnded) {
    return "EntityMovementTimerEnded";
  }
  if (eventCode == EventCode::kEntityPositionUpdated) {
    return "EntityPositionUpdated";
  }
  if (eventCode == EventCode::kEntityNotMovingAngleChanged) {
    return "EntityNotMovingAngleChanged";
  }
  if (eventCode == EventCode::kEntityBodyStateChanged) {
    return "EntityBodyStateChanged";
  }
  if (eventCode == EventCode::kEntityLifeStateChanged) {
    return "EntityLifeStateChanged";
  }
  if (eventCode == EventCode::kEntityEnteredGeometry) {
    return "EntityEnteredGeometry";
  }
  if (eventCode == EventCode::kEntityExitedGeometry) {
    return "EntityExitedGeometry";
  }
  if (eventCode == EventCode::kTrainingAreaSet) {
    return "TrainingAreaSet";
  }
  if (eventCode == EventCode::kTrainingAreaReset) {
    return "TrainingAreaReset";
  }
  if (eventCode == EventCode::kKnockedBack) {
    return "KnockedBack";
  }
  if (eventCode == EventCode::kKnockedDown) {
    return "KnockedDown";
  }
  if (eventCode == EventCode::kKnockbackStunEnded) {
    return "KnockbackStunEnded";
  }
  if (eventCode == EventCode::kKnockdownStunEnded) {
    return "KnockdownStunEnded";
  }
  if (eventCode == EventCode::kMovementRequestTimedOut) {
    return "MovementRequestTimedOut";
  }
  if (eventCode == EventCode::kWalkingPathUpdated) {
    return "WalkingPathUpdated";
  }
  if (eventCode == EventCode::kNewConfigReceived) {
    return "NewConfigReceived";
  }
  if (eventCode == EventCode::kConfigUpdated) {
    return "ConfigUpdated";
  }
  if (eventCode == EventCode::kInventoryItemUpdated) {
    return "InventoryItemUpdated";
  }
  if (eventCode == EventCode::kHwanPointsUpdated) {
    return "HwanPointsUpdated";
  }
  if (eventCode == EventCode::kAlchemyCompleted) {
    return "AlchemyCompleted";
  }
  if (eventCode == EventCode::kAlchemyTimedOut) {
    return "AlchemyTimedOut";
  }
  if (eventCode == EventCode::kGmCommandTimedOut) {
    return "GmCommandTimedOut";
  }
  if (eventCode == EventCode::kChatReceived) {
    return "ChatReceived";
  }
  if (eventCode == EventCode::kGameReset) {
    return "GameReset";
  }
  if (eventCode == EventCode::kSetCurrentPositionAsTrainingCenter) {
    return "SetCurrentPositionAsTrainingCenter";
  }
  if (eventCode == EventCode::kResurrectOption) {
    return "ResurrectOption";
  }
  if (eventCode == EventCode::kLearnMasterySuccess) {
    return "LearnMasterySuccess";
  }
  if (eventCode == EventCode::kLearnSkillSuccess) {
    return "LearnSkillSuccess";
  }
  if (eventCode == EventCode::kTimeout) {
    return "Timeout";
  }
  if (eventCode == EventCode::kStateUpdated) {
    return "StateUpdated";
  }
  if (eventCode == EventCode::kStateShardIdUpdated) {
    return "StateShardIdUpdated";
  }
  if (eventCode == EventCode::kStateConnectedToAgentServerUpdated) {
    return "StateConnectedToAgentServerUpdated";
  }
  if (eventCode == EventCode::kStateReceivedCaptchaPromptUpdated) {
    return "StateReceivedCaptchaPromptUpdated";
  }
  if (eventCode == EventCode::kStateCharacterListUpdated) {
    return "StateCharacterListUpdated";
  }
  LOG(WARNING) << absl::StreamFormat("Asking for string for unknown event %d", static_cast<int>(eventCode));
  return "UNKNOWN";
}

Event::Event(EventId id, EventCode code) :
    eventId(id), eventCode(code) {}

SkillCooldownEnded::SkillCooldownEnded(EventId id, sro::scalar_types::ReferenceSkillId skillId) :
    Event(id, EventCode::kSkillCooldownEnded), skillRefId(skillId) {}

InventoryUpdated::InventoryUpdated(EventId id, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) :
    Event(id, EventCode::kInventoryUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

AvatarInventoryUpdated::AvatarInventoryUpdated(EventId id, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) :
    Event(id, EventCode::kAvatarInventoryUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

CosInventoryUpdated::CosInventoryUpdated(EventId id, sro::scalar_types::EntityGlobalId gId, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) :
    Event(id, EventCode::kCosInventoryUpdated), globalId(gId), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

StorageUpdated::StorageUpdated(EventId id, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) :
    Event(id, EventCode::kStorageUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

GuildStorageUpdated::GuildStorageUpdated(EventId id, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) :
    Event(id, EventCode::kGuildStorageUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

ItemUseFailed::ItemUseFailed(EventId id, uint8_t slotNum, type_id::TypeId typeId, packet::enums::InventoryErrorCode reason_param) :
    Event(id, EventCode::kItemUseFailed), inventorySlotNum(slotNum), itemTypeId(typeId), reason(reason_param) {}

InjectPacket::InjectPacket(EventId id, InjectPacket::Direction dir, uint16_t op, const std::string &d) :
    Event(id, EventCode::kInjectPacket), direction(dir), opcode(op), data(d) {}

CosSpawned::CosSpawned(EventId id, sro::scalar_types::EntityGlobalId cosGId) :
    Event(id, EventCode::kCosSpawned), cosGlobalId(cosGId) {}

EntitySpawned::EntitySpawned(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntitySpawned), globalId(globalId) {}

EntityDespawned::EntityDespawned(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityDespawned), globalId(globalId) {}

EntityMovementEnded::EntityMovementEnded(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityMovementEnded), globalId(globalId) {}

EntityMovementBegan::EntityMovementBegan(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityMovementBegan), globalId(globalId) {}

EntityMovementTimerEnded::EntityMovementTimerEnded(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityMovementTimerEnded), globalId(globalId) {}

EntityPositionUpdated::EntityPositionUpdated(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityPositionUpdated), globalId(globalId) {}

EntityNotMovingAngleChanged::EntityNotMovingAngleChanged(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityNotMovingAngleChanged), globalId(globalId) {}

EntityBodyStateChanged::EntityBodyStateChanged(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityBodyStateChanged), globalId(globalId) {}

EntityLifeStateChanged::EntityLifeStateChanged(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityLifeStateChanged), globalId(globalId) {}

EntityEnteredGeometry::EntityEnteredGeometry(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityEnteredGeometry), globalId(globalId) {}

EntityExitedGeometry::EntityExitedGeometry(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityExitedGeometry), globalId(globalId) {}

SkillBegan::SkillBegan(EventId id, sro::scalar_types::EntityGlobalId casterId, sro::scalar_types::ReferenceObjectId skillId) :
    Event(id, EventCode::kSkillBegan), casterGlobalId(casterId), skillRefId(skillId) {}

SkillEnded::SkillEnded(EventId id, sro::scalar_types::EntityGlobalId casterId, sro::scalar_types::ReferenceObjectId skillId) :
    Event(id, EventCode::kSkillEnded), casterGlobalId(casterId), skillRefId(skillId) {}

DealtDamage::DealtDamage(EventId id, sro::scalar_types::EntityGlobalId sourceId, sro::scalar_types::EntityGlobalId targetId, uint32_t damageAmount) :
    Event(id, EventCode::kDealtDamage), sourceId(sourceId), targetId(targetId), damageAmount(damageAmount) {}

KilledEntity::KilledEntity(EventId id, sro::scalar_types::EntityGlobalId targetId) :
    Event(id, EventCode::kKilledEntity), targetId(targetId) {}

OurSkillFailed::OurSkillFailed(EventId id, sro::scalar_types::ReferenceSkillId skillId, uint16_t err) :
    Event(id, EventCode::kOurSkillFailed), skillRefId(skillId), errorCode(err) {}

EntityHpChanged::EntityHpChanged(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityHpChanged), globalId(globalId) {}

BuffAdded::BuffAdded(EventId id, sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceObjectId buffId) :
    Event(id, EventCode::kPlayerCharacterBuffAdded), entityGlobalId(entityId), buffRefId(buffId) {}

BuffRemoved::BuffRemoved(EventId id, sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceObjectId buffId) :
    Event(id, EventCode::kPlayerCharacterBuffRemoved), entityGlobalId(entityId), buffRefId(buffId) {}

CommandError::CommandError(EventId id, const packet::structures::ActionCommand &cmd) :
    Event(id, EventCode::kOurCommandError), command(cmd) {}

ItemUseTimeout::ItemUseTimeout(EventId id, uint8_t slot, type_id::TypeId tid) :
    Event(id, EventCode::kItemUseTimeout), slotNum(slot), typeData(tid) {}

SkillCastTimeout::SkillCastTimeout(EventId id, sro::scalar_types::ReferenceObjectId skillId) :
    Event(id, EventCode::kSkillCastTimeout), skillId(skillId) {}

EntityOwnershipRemoved::EntityOwnershipRemoved(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityOwnershipRemoved), globalId(globalId) {}

StateMachineCreated::StateMachineCreated(EventId id, const std::string &name) :
    Event(id, EventCode::kStateMachineCreated), stateMachineName(name) {}

ItemCooldownEnded::ItemCooldownEnded(EventId id, type_id::TypeId tId) :
    Event(id, EventCode::kItemCooldownEnded), typeId(tId) {}

WalkingPathUpdated::WalkingPathUpdated(EventId id, const std::vector<packet::building::NetworkReadyPosition> &waypoints) :
    Event(id, EventCode::kWalkingPathUpdated), waypoints(waypoints) {}

NewConfigReceived::NewConfigReceived(EventId id, const proto::config::Config &config_param) :
    Event(id, EventCode::kNewConfigReceived), config(config_param) {}

InventoryItemUpdated::InventoryItemUpdated(EventId id, const uint8_t &slot) :
    Event(id, EventCode::kInventoryItemUpdated), slotIndex(slot) {}

ChatReceived::ChatReceived(EventId id, packet::enums::ChatType type, sro::scalar_types::EntityGlobalId senderGlobalId, const std::string &msg) :
    Event(id, EventCode::kChatReceived), chatType(type), sender(senderGlobalId), message(msg) {}
ChatReceived::ChatReceived(EventId id, packet::enums::ChatType type, const std::string &senderName, const std::string &msg) :
    Event(id, EventCode::kChatReceived), chatType(type), sender(senderName), message(msg) {}

ConfigUpdated::ConfigUpdated(EventId id, const proto::config::Config &c) :
    Event(id, EventCode::kConfigUpdated), config(c) {}

ResurrectOption::ResurrectOption(EventId id, packet::enums::ResurrectionOptionFlag option) :
    Event(id, EventCode::kResurrectOption), option(option) {}

LearnMasterySuccess::LearnMasterySuccess(EventId id, sro::scalar_types::ReferenceMasteryId masteryId) :
    Event(id, EventCode::kLearnMasterySuccess), masteryId(masteryId) {}

LearnSkillSuccess::LearnSkillSuccess(EventId id, sro::scalar_types::ReferenceSkillId newSkillId, std::optional<sro::scalar_types::ReferenceSkillId> oldSkillId) :
    Event(id, EventCode::kLearnSkillSuccess), newSkillRefId(newSkillId), oldSkillRefId(oldSkillId) {}

} // namespace event