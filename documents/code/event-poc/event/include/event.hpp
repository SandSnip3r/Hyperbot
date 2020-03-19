#ifndef EVENT_HPP
#define EVENT_HPP

namespace event {

enum class EventCode {
  kTest1 = 1,
  kTest2 = 2
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

#endif // EVENT_HPP