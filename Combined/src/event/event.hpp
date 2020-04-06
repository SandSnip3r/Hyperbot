#ifndef EVENT_EVENT_HPP_
#define EVENT_EVENT_HPP_

namespace event {

enum class EventCode {
  kHpPotionCooldownEnded,
  kMpPotionCooldownEnded,
  kVigorPotionCooldownEnded,
  kUniversalPillCooldownEnded,
  kPurificationPillCooldownEnded,
  kHpPercentChanged,
  kMpPercentChanged,
  kStatesChanged
};

class Event {
public:
  explicit Event(EventCode eventCode);
  EventCode getEventCode() const;
  virtual ~Event() = default;
private:
  EventCode eventCode_;
};

} // namespace event

#endif // EVENT_EVENT_HPP_