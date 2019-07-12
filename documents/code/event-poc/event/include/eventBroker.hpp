#ifndef EVENT_BROKER_HPP
#define EVENT_BROKER_HPP

#include "event.hpp"
#include "timerManager.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <unordered_map>

namespace event {

class EventBroker {
private:
  using EventHandleFunction = std::function<void(const std::unique_ptr<Event>&)>;
  using PacketSubscriptionMap = std::unordered_map<EventCode, std::vector<EventHandleFunction>>;
  PacketSubscriptionMap subscriptions_;
public:
  void run();
  bool publishEvent(std::unique_ptr<Event> event);
  
  //TODO: return timer Id
  bool publishDelayedEvent(std::unique_ptr<Event> event, std::chrono::milliseconds delay);

  //TODO: return subscription Id
  void subscribeToEvent(EventCode eventCode, EventHandleFunction &&handleFunc);
private:
  TimerManager timerManager_;
  void timerFinished(Event *event);
};

} // namespace event

#endif // EVENT_BROKER_HPP