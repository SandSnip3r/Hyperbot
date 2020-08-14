#include "event.hpp"

namespace event {

Event::Event(EventCode event) : eventCode(event) {}

SkillCooldownEnded::SkillCooldownEnded(int32_t skillId) : Event(EventCode::kSkillCooldownEnded), skillRefId(skillId) {}

InventorySlotUpdated::InventorySlotUpdated(int8_t slot) : Event(EventCode::kInventorySlotUpdated), slotNum(slot) {}

} // namespace event