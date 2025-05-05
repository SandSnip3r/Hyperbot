#include "event.hpp"

#include <absl/log/log.h>

namespace event {

Event::Event(EventId id, EventCode code) :
    eventId(id), eventCode(code) {}

SessionSpecificEvent::SessionSpecificEvent(EventId id, EventCode code, SessionId sessionId) :
    Event(id, code), sessionId(sessionId) {}

ServerAuthSuccess::ServerAuthSuccess(EventId id, SessionId sessionId) :
    SessionSpecificEvent(id, EventCode::kServerAuthSuccess, sessionId) {}

GatewayPatchResponseReceived::GatewayPatchResponseReceived(EventId id, SessionId sessionId) :
    SessionSpecificEvent(id, EventCode::kGatewayPatchResponseReceived, sessionId) {}

ShardListReceived::ShardListReceived(EventId id, SessionId sessionId, const std::vector<packet::structures::Shard> &shards) :
    SessionSpecificEvent(id, EventCode::kShardListReceived, sessionId), shards(shards) {}

IbuvChallengeReceived::IbuvChallengeReceived(EventId id, SessionId sessionId) :
    SessionSpecificEvent(id, EventCode::kIbuvChallengeReceived, sessionId) {}

GatewayLoginResponseReceived::GatewayLoginResponseReceived(EventId id, SessionId sessionId, uint32_t agentServerToken) :
    SessionSpecificEvent(id, EventCode::kGatewayLoginResponseReceived, sessionId), agentServerToken(agentServerToken) {}

ConnectedToAgentServer::ConnectedToAgentServer(EventId id, SessionId sessionId) :
    SessionSpecificEvent(id, EventCode::kConnectedToAgentServer, sessionId) {}

CharacterListReceived::CharacterListReceived(EventId id, SessionId sessionId, const std::vector<packet::structures::character_selection::Character> &characters) :
    SessionSpecificEvent(id, EventCode::kCharacterListReceived, sessionId), characters(characters) {}

CharacterSelectionJoinSuccess::CharacterSelectionJoinSuccess(EventId id, SessionId sessionId) :
    SessionSpecificEvent(id, EventCode::kCharacterSelectionJoinSuccess, sessionId) {}

SelfSpawned::SelfSpawned(EventId id, SessionId sessionId, sro::scalar_types::EntityGlobalId globalId) :
    SessionSpecificEvent(id, EventCode::kSelfSpawned, sessionId), globalId(globalId) {}

InternalSkillCooldownEnded::InternalSkillCooldownEnded(EventId id, sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceSkillId skillId) :
    Event(id, EventCode::kInternalSkillCooldownEnded), globalId(globalId), skillRefId(skillId) {}

SkillCooldownEnded::SkillCooldownEnded(EventId id, sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::ReferenceSkillId skillId) :
    Event(id, EventCode::kSkillCooldownEnded), globalId(globalId), skillRefId(skillId) {}

ItemMoved::ItemMoved(EventId id, sro::scalar_types::EntityGlobalId globalId, std::optional<sro::storage::Position> source, std::optional<sro::storage::Position> destination) :
    Event(id, EventCode::kItemMoved), globalId(globalId), source(source), destination(destination) {}

ItemMoveFailed::ItemMoveFailed(EventId id, sro::scalar_types::EntityGlobalId globalId, uint16_t errorCode) :
    Event(id, EventCode::kItemMoveFailed), globalId(globalId), errorCode(errorCode) {}

ItemUseSuccess::ItemUseSuccess(EventId id, sro::scalar_types::EntityGlobalId globalId, sro::scalar_types::StorageIndexType slotNum, sro::scalar_types::ReferenceObjectId refId) :
    Event(id, EventCode::kItemUseSuccess), globalId(globalId), slotNum(slotNum), refId(refId) {}

ItemUseFailed::ItemUseFailed(EventId id, uint8_t slotNum, type_id::TypeId typeId, packet::enums::InventoryErrorCode reason_param) :
    Event(id, EventCode::kItemUseFailed), inventorySlotNum(slotNum), itemTypeId(typeId), reason(reason_param) {}

ItemUseTimeout::ItemUseTimeout(EventId id, uint8_t slot, type_id::TypeId tid) :
    Event(id, EventCode::kItemUseTimeout), slotNum(slot), typeData(tid) {}

InjectPacket::InjectPacket(EventId id, InjectPacket::Direction dir, uint16_t op, const std::string &d) :
    Event(id, EventCode::kInjectPacket), direction(dir), opcode(op), data(d) {}

CosSpawned::CosSpawned(EventId id, sro::scalar_types::EntityGlobalId cosGId) :
    Event(id, EventCode::kCosSpawned), cosGlobalId(cosGId) {}

EnteredNewRegion::EnteredNewRegion(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEnteredNewRegion), globalId(globalId) {}

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

SkillFailed::SkillFailed(EventId id, sro::scalar_types::EntityGlobalId casterGlobalId, sro::scalar_types::ReferenceSkillId skillId, uint16_t err) :
    Event(id, EventCode::kSkillFailed), casterGlobalId(casterGlobalId), skillRefId(skillId), errorCode(err) {}

EntityHpChanged::EntityHpChanged(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityHpChanged), globalId(globalId) {}

EntityMpChanged::EntityMpChanged(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityMpChanged), globalId(globalId) {}

EntityStatesChanged::EntityStatesChanged(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityStatesChanged), globalId(globalId) {}

BuffAdded::BuffAdded(EventId id, sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceSkillId buffId) :
    Event(id, EventCode::kPlayerCharacterBuffAdded), entityGlobalId(entityId), buffRefId(buffId) {}

BuffRemoved::BuffRemoved(EventId id, sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceSkillId buffId) :
    Event(id, EventCode::kPlayerCharacterBuffRemoved), entityGlobalId(entityId), buffRefId(buffId) {}

CommandError::CommandError(EventId id, sro::scalar_types::EntityGlobalId issuingGlobalId, const packet::structures::ActionCommand &cmd) :
    Event(id, EventCode::kCommandError), issuingGlobalId(issuingGlobalId), command(cmd) {}

CommandSkipped::CommandSkipped(EventId id, sro::scalar_types::EntityGlobalId issuingGlobalId, const packet::structures::ActionCommand &cmd) :
    Event(id, EventCode::kCommandSkipped), issuingGlobalId(issuingGlobalId), command(cmd) {}

SkillCastTimeout::SkillCastTimeout(EventId id, sro::scalar_types::ReferenceObjectId skillId) :
    Event(id, EventCode::kSkillCastTimeout), skillId(skillId) {}

EntityOwnershipRemoved::EntityOwnershipRemoved(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEntityOwnershipRemoved), globalId(globalId) {}

StateMachineCreated::StateMachineCreated(EventId id, const std::string &name) :
    Event(id, EventCode::kStateMachineCreated), stateMachineName(name) {}

InternalItemCooldownEnded::InternalItemCooldownEnded(EventId eventId, sro::scalar_types::EntityGlobalId globalId, type_id::TypeId typeId) :
    Event(eventId, EventCode::kInternalItemCooldownEnded), globalId(globalId), typeId(typeId) {}

ItemCooldownEnded::ItemCooldownEnded(EventId eventId, sro::scalar_types::EntityGlobalId globalId, type_id::TypeId typeId) :
    Event(eventId, EventCode::kItemCooldownEnded), globalId(globalId), typeId(typeId) {}

WalkingPathUpdated::WalkingPathUpdated(EventId id, const std::vector<packet::building::NetworkReadyPosition> &waypoints) :
    Event(id, EventCode::kWalkingPathUpdated), waypoints(waypoints) {}

InventoryItemUpdated::InventoryItemUpdated(EventId id, const uint8_t &slot) :
    Event(id, EventCode::kInventoryItemUpdated), slotIndex(slot) {}

ChatReceived::ChatReceived(EventId id, packet::enums::ChatType type, sro::scalar_types::EntityGlobalId senderGlobalId, const std::string &msg) :
    Event(id, EventCode::kChatReceived), chatType(type), sender(senderGlobalId), message(msg) {}
ChatReceived::ChatReceived(EventId id, packet::enums::ChatType type, const std::string &senderName, const std::string &msg) :
    Event(id, EventCode::kChatReceived), chatType(type), sender(senderName), message(msg) {}

ResurrectOption::ResurrectOption(EventId id, packet::enums::ResurrectionOptionFlag option) :
    Event(id, EventCode::kResurrectOption), option(option) {}

LearnMasterySuccess::LearnMasterySuccess(EventId id, sro::scalar_types::ReferenceMasteryId masteryId) :
    Event(id, EventCode::kLearnMasterySuccess), masteryId(masteryId) {}

LearnSkillSuccess::LearnSkillSuccess(EventId id, sro::scalar_types::ReferenceSkillId newSkillId, std::optional<sro::scalar_types::ReferenceSkillId> oldSkillId) :
    Event(id, EventCode::kLearnSkillSuccess), newSkillRefId(newSkillId), oldSkillRefId(oldSkillId) {}

OperatorRequestSuccess::OperatorRequestSuccess(EventId id, sro::scalar_types::EntityGlobalId globalId, packet::enums::OperatorCommand operatorCommand) :
    Event(id, EventCode::kOperatorRequestSuccess), globalId(globalId), operatorCommand(operatorCommand) {}

OperatorRequestError::OperatorRequestError(EventId id, sro::scalar_types::EntityGlobalId globalId, packet::enums::OperatorCommand operatorCommand) :
    Event(id, EventCode::kOperatorRequestError), globalId(globalId), operatorCommand(operatorCommand) {}

EquipCountdownStart::EquipCountdownStart(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kEquipCountdownStart), globalId(globalId) {}

FreePvpUpdateSuccess::FreePvpUpdateSuccess(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kFreePvpUpdateSuccess), globalId(globalId) {}

PvpManagerReadyForAssignment::PvpManagerReadyForAssignment(EventId id, SessionId sessionId) :
    Event(id, EventCode::kPvpManagerReadyForAssignment), sessionId(sessionId) {}

BeginPvp::BeginPvp(EventId id, common::PvpDescriptor pvpDescriptor) :
    Event(id, EventCode::kBeginPvp), pvpDescriptor(pvpDescriptor) {}

ReadyForPvp::ReadyForPvp(EventId id, sro::scalar_types::EntityGlobalId globalId) :
    Event(id, EventCode::kReadyForPvp), globalId(globalId) {}

ClientDied::ClientDied(EventId id, ClientManagerInterface::ClientId clientId) :
    Event(id, EventCode::kClientDied), clientId(clientId) {}

RlUiSaveCheckpoint::RlUiSaveCheckpoint(EventId id, const std::string &checkpointName) :
    Event(id, EventCode::kRlUiSaveCheckpoint), checkpointName(checkpointName) {}

RlUiLoadCheckpoint::RlUiLoadCheckpoint(EventId id, const std::string &checkpointName) :
    Event(id, EventCode::kRlUiLoadCheckpoint), checkpointName(checkpointName) {}

RlUiDeleteCheckpoints::RlUiDeleteCheckpoints(EventId id, const std::vector<std::string> &checkpointNames) :
    Event(id, EventCode::kRlUiDeleteCheckpoints), checkpointNames(checkpointNames) {}

} // namespace event