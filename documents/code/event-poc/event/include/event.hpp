#ifndef EVENT_HPP
#define EVENT_HPP

namespace event {

enum class EventCode {};

class Event {
public:
  explicit Event(EventCode eventCode);
  EventCode getEventCode() const;
  virtual ~Event() = 0;
private:
  EventCode eventCode_;
};

} // namespace event

#endif // EVENT_HPP