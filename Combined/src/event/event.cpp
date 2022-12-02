#include "event.hpp"

namespace event {

Event::Event(EventCode event) : eventCode(event) {}

SkillCooldownEnded::SkillCooldownEnded(int32_t skillId) : Event(EventCode::kSkillCooldownEnded), skillRefId(skillId) {}

InventoryUpdated::InventoryUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kInventoryUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

AvatarInventoryUpdated::AvatarInventoryUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kAvatarInventoryUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

CosInventoryUpdated::CosInventoryUpdated(uint32_t gId, const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kCosInventoryUpdated), globalId(gId), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

StorageUpdated::StorageUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kStorageUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

GuildStorageUpdated::GuildStorageUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kGuildStorageUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

ItemWaitForReuseDelay::ItemWaitForReuseDelay(uint8_t slotNum, type_id::TypeId typeId) : Event(EventCode::kItemWaitForReuseDelay), inventorySlotNum(slotNum), itemTypeId(typeId) {}

InjectPacket::InjectPacket(InjectPacket::Direction dir, uint16_t op, const std::string &d) : Event(EventCode::kInjectPacket), direction(dir), opcode(op), data(d) {}

CosSpawned::CosSpawned(uint32_t cosGId) : Event(EventCode::kCosSpawned), cosGlobalId(cosGId) {}

EntitySpawned::EntitySpawned(uint32_t id) : Event(EventCode::kEntitySpawned), globalId(id) {}

EntityDespawned::EntityDespawned(uint32_t id) : Event(EventCode::kEntityDespawned), globalId(id) {}

EntityMovementEnded::EntityMovementEnded(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityMovementEnded), globalId(id) {}

EntityMovementBegan::EntityMovementBegan(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityMovementBegan), globalId(id) {}

EntityMovementTimerEnded::EntityMovementTimerEnded(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityMovementTimerEnded), globalId(id) {}

EntityPositionUpdated::EntityPositionUpdated(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityPositionUpdated), globalId(id) {}

EntityNotMovingAngleChanged::EntityNotMovingAngleChanged(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityNotMovingAngleChanged), globalId(id) {}

EntityLifeStateChanged::EntityLifeStateChanged(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityLifeStateChanged), globalId(id) {}

EntityEnteredGeometry::EntityEnteredGeometry(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityEnteredGeometry), globalId(id) {}

EntityExitedGeometry::EntityExitedGeometry(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityExitedGeometry), globalId(id) {}

SkillBegan::SkillBegan(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kSkillBegan), casterGlobalId(id) {}

SkillEnded::SkillEnded(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kSkillEnded), casterGlobalId(id) {}

EntityHpChanged::EntityHpChanged(sro::scalar_types::EntityGlobalId id) : Event(EventCode::kEntityHpChanged), globalId(id) {}

CommandError::CommandError(const packet::structures::ActionCommand &cmd) : Event(EventCode::kOurCommandError), command(cmd) {}

ItemUseTimeout::ItemUseTimeout(uint8_t slot, type_id::TypeId tid) : Event(EventCode::kItemUseTimeout), slotNum(slot), typeData(tid) {}

} // namespace event