#include "event.hpp"

namespace event {

Event::Event(EventCode event) : eventCode(event) {}

SkillCooldownEnded::SkillCooldownEnded(int32_t skillId) : Event(EventCode::kSkillCooldownEnded), skillRefId(skillId) {}

InventoryUpdated::InventoryUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kInventoryUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

AvatarInventoryUpdated::AvatarInventoryUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kAvatarInventoryUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

CosInventoryUpdated::CosInventoryUpdated(uint32_t gId, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kCosInventoryUpdated), globalId(gId), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

StorageUpdated::StorageUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kStorageUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

GuildStorageUpdated::GuildStorageUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kGuildStorageUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

ItemUseFailed::ItemUseFailed(uint8_t slotNum, type_id::TypeId typeId, packet::enums::InventoryErrorCode reason_param) : Event(EventCode::kItemUseFailed), inventorySlotNum(slotNum), itemTypeId(typeId), reason(reason_param) {}

InjectPacket::InjectPacket(InjectPacket::Direction dir, uint16_t op, const std::string &d) : Event(EventCode::kInjectPacket), direction(dir), opcode(op), data(d) {}

CosSpawned::CosSpawned(uint32_t cosGId) : Event(EventCode::kCosSpawned), cosGlobalId(cosGId) {}

EntitySpawned::EntitySpawned(uint32_t id) : Event(EventCode::kEntitySpawned), globalId(id) {}

EntityDespawned::EntityDespawned(uint32_t id) : Event(EventCode::kEntityDespawned), globalId(id) {}

EntityMovementEnded::EntityMovementEnded(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityMovementEnded), globalId(id) {}

EntityMovementBegan::EntityMovementBegan(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityMovementBegan), globalId(id) {}

EntityMovementTimerEnded::EntityMovementTimerEnded(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityMovementTimerEnded), globalId(id) {}

EntityPositionUpdated::EntityPositionUpdated(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityPositionUpdated), globalId(id) {}

EntityNotMovingAngleChanged::EntityNotMovingAngleChanged(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityNotMovingAngleChanged), globalId(id) {}

EntityBodyStateChanged::EntityBodyStateChanged(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityBodyStateChanged), globalId(id) {}

EntityLifeStateChanged::EntityLifeStateChanged(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityLifeStateChanged), globalId(id) {}

EntityEnteredGeometry::EntityEnteredGeometry(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityEnteredGeometry), globalId(id) {}

EntityExitedGeometry::EntityExitedGeometry(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityExitedGeometry), globalId(id) {}

SkillBegan::SkillBegan(sro::scalar_types::EntityGlobalId casterId, sro::scalar_types::ReferenceObjectId skillId) : Event(EventCode::kSkillBegan), casterGlobalId(casterId), skillRefId(skillId) {}

SkillEnded::SkillEnded(sro::scalar_types::EntityGlobalId casterId, sro::scalar_types::ReferenceObjectId skillId) : Event(EventCode::kSkillEnded), casterGlobalId(casterId), skillRefId(skillId) {}

DealtDamage::DealtDamage(sro::scalar_types::EntityGlobalId targetId, uint32_t damageAmount) : Event(EventCode::kDealtDamage), targetId(targetId), damageAmount(damageAmount) {}

KilledEntity::KilledEntity(sro::scalar_types::EntityGlobalId targetId) : Event(EventCode::kKilledEntity), targetId(targetId) {}

OurSkillFailed::OurSkillFailed(sro::scalar_types::ReferenceObjectId id, uint16_t err) : Event(EventCode::kOurSkillFailed), skillRefId(id), errorCode(err) {}

EntityHpChanged::EntityHpChanged(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityHpChanged), globalId(id) {}

BuffAdded::BuffAdded(sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceObjectId buffId) : Event(EventCode::kPlayerCharacterBuffAdded), entityGlobalId(entityId), buffRefId(buffId) {}

BuffRemoved::BuffRemoved(sro::scalar_types::EntityGlobalId entityId, sro::scalar_types::ReferenceObjectId buffId) : Event(EventCode::kPlayerCharacterBuffRemoved), entityGlobalId(entityId), buffRefId(buffId) {}

CommandError::CommandError(const packet::structures::ActionCommand &cmd) : Event(EventCode::kOurCommandError), command(cmd) {}

ItemUseTimeout::ItemUseTimeout(uint8_t slot, type_id::TypeId tid) : Event(EventCode::kItemUseTimeout), slotNum(slot), typeData(tid) {}

SkillCastTimeout::SkillCastTimeout(sro::scalar_types::ReferenceObjectId skillId) : Event(EventCode::kSkillCastTimeout), skillId(skillId) {}

EntityOwnershipRemoved::EntityOwnershipRemoved(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityOwnershipRemoved), globalId(id) {}

StateMachineCreated::StateMachineCreated(const std::string &name) : Event(EventCode::kStateMachineCreated), stateMachineName(name) {}

ItemCooldownEnded::ItemCooldownEnded(type_id::TypeId tId) : Event(EventCode::kItemCooldownEnded), typeId(tId) {}

WalkingPathUpdated::WalkingPathUpdated(const std::vector<packet::building::NetworkReadyPosition> &waypoints) : Event(EventCode::kWalkingPathUpdated), waypoints(waypoints) {}

NewConfigReceived::NewConfigReceived(const proto::config::Config &config_param) : Event(EventCode::kNewConfigReceived), config(config_param) {}

InventoryItemUpdated::InventoryItemUpdated(const uint8_t &slot) : Event(EventCode::kInventoryItemUpdated), slotIndex(slot) {}

ChatReceived::ChatReceived(packet::enums::ChatType type, uint32_t senderGlobalId, const std::string &msg) : Event(EventCode::kChatReceived), chatType(type), sender(senderGlobalId), message(msg) {}
ChatReceived::ChatReceived(packet::enums::ChatType type, const std::string &senderName, const std::string &msg) : Event(EventCode::kChatReceived), chatType(type), sender(senderName), message(msg) {}

} // namespace event