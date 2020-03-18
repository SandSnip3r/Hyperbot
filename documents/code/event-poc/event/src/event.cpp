#include "../include/event.hpp"

namespace event {

Event::Event(EventCode eventCode) : eventCode_(eventCode) {}

EventCode Event::getEventCode() const {
  return eventCode_;
}

// Event::~Event() {}

} // namespace event