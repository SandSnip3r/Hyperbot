#ifndef EVENT_EVENT_BROKER_HPP_
#define EVENT_EVENT_BROKER_HPP_

#include "event/event.hpp"
#include "timerManager.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

template<typename T, typename... U>
size_t getAddress(std::function<T(U...)> f) {
  // TODO: This crashes us
  // It came from here https://stackoverflow.com/questions/18039723/c-trying-to-get-function-address-from-a-stdfunction
  typedef T(fnType)(U...);
  fnType ** fnPointer = f.template target<fnType*>();
  return (size_t) *fnPointer;
}

namespace broker {

class EventBroker {
public:
  using DelayedEventId = TimerManager::TimerId;
  using SubscriptionId = int;
private:
  using EventHandleFunction = std::function<void(const event::Event*)>;
  struct EventSubscription {
    EventSubscription(SubscriptionId subId, EventHandleFunction &&func) : id(subId), handleFunction(func) {}
    SubscriptionId id;
    EventHandleFunction handleFunction;
  };
  using PacketSubscriptionMap = std::unordered_map<event::EventCode, std::vector<EventSubscription>>;
public:
  void runAsync();

  template<typename EventType, typename... Args>
  void publishEvent(Args&&... args) {
    publishEvent(std::make_unique<EventType>(std::forward<Args>(args)...));
  }
  void publishEvent(event::EventCode eventCode);

  template<typename EventType, typename... Args>
  EventBroker::DelayedEventId publishDelayedEvent(std::chrono::milliseconds delay, Args&&... args) {
    return publishDelayedEvent(delay, std::make_unique<EventType>(std::forward<Args>(args)...));
  }
  EventBroker::DelayedEventId publishDelayedEvent(std::chrono::milliseconds delay, event::EventCode eventCode);

  bool cancelDelayedEvent(DelayedEventId id);
  SubscriptionId subscribeToEvent(event::EventCode eventCode, EventHandleFunction &&handleFunc);
  void unsubscribeFromEvent(SubscriptionId id);
private:
  int subscriptionIdCounter_{0};
  PacketSubscriptionMap subscriptions_;
  std::mutex subscriptionMutex_;
  TimerManager timerManager_;
  void publishEvent(std::unique_ptr<event::Event> event);
  DelayedEventId publishDelayedEvent(std::chrono::milliseconds delay, std::unique_ptr<event::Event> event);

  void notifySubscribers(std::unique_ptr<event::Event> event);
  void timerFinished(event::Event *event);
};

} // namespace broker

#endif // EVENT_EVENT_BROKER_HPP_