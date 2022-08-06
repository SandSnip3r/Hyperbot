#include "event.hpp"

namespace event {

Event::Event(EventCode event) : eventCode(event) {}

SkillCooldownEnded::SkillCooldownEnded(int32_t skillId) : Event(EventCode::kSkillCooldownEnded), skillRefId(skillId) {}

InventoryUpdated::InventoryUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kInventoryUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

StorageUpdated::StorageUpdated(const std::optional<int8_t> &srcSlot, const std::optional<int8_t> &destSlot) : Event(EventCode::kStorageUpdated), srcSlotNum(srcSlot), destSlotNum(destSlot) {}

ItemWaitForReuseDelay::ItemWaitForReuseDelay(uint8_t slotNum, uint16_t typeId) : Event(EventCode::kItemWaitForReuseDelay), inventorySlotNum(slotNum), itemTypeId(typeId) {}

InjectPacket::InjectPacket(InjectPacket::Direction dir, uint16_t op, const std::string &d) : Event(EventCode::kInjectPacket), direction(dir), opcode(op), data(d) {}

} // namespace event