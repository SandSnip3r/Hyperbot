#ifndef EVENT_EVENT_HPP_
#define EVENT_EVENT_HPP_

#include <cstdint>

namespace event {

enum class EventCode {
  kSpawned,
  kHpPotionCooldownEnded,
  kMpPotionCooldownEnded,
  kVigorPotionCooldownEnded,
  kUniversalPillCooldownEnded,
  kPurificationPillCooldownEnded,
  kHpPercentChanged,
  kMpPercentChanged,
  kStatesChanged,
  kSkillCooldownEnded,
  kInventorySlotUpdated,
  kSkillCastAboutToEnd,
  kKnockbackStatusEnded,
  kMovementEnded,
  kCharacterSpeedUpdated,
  kDropGold,
  kRepublish
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

struct InventorySlotUpdated : public Event {
public:
  InventorySlotUpdated(int8_t slot);
  const int8_t slotNum;
  virtual ~InventorySlotUpdated() = default;
};

struct DropGold : public Event {
public:
  DropGold(int amount, int count);
  const int goldAmount, goldDropCount;
  virtual ~DropGold() = default;
};

} // namespace event

#endif // EVENT_EVENT_HPP_