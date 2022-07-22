#include "event.hpp"

namespace event {

Event::Event(EventCode event) : eventCode(event) {}

SkillCooldownEnded::SkillCooldownEnded(int32_t skillId) : Event(EventCode::kSkillCooldownEnded), skillRefId(skillId) {}

InventorySlotUpdated::InventorySlotUpdated(int8_t slot) : Event(EventCode::kInventorySlotUpdated), slotNum(slot) {}

DropGold::DropGold(int amount, int count) : Event(EventCode::kDropGold), goldAmount(amount), goldDropCount(count) {}

ItemWaitForReuseDelay::ItemWaitForReuseDelay(uint8_t slotNum, uint16_t typeId) : Event(EventCode::kItemWaitForReuseDelay), inventorySlotNum(slotNum), itemTypeId(typeId) {}

} // namespace event