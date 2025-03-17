#ifndef BROKER_EVENT_BROKER_HPP_
#define BROKER_EVENT_BROKER_HPP_

#include "event/event.hpp"
#include "timerManager.hpp"

#include <tracy/Tracy.hpp>

#include <absl/container/flat_hash_map.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

template<typename T, typename... U>
size_t getAddress(std::function<T(U...)> f) {
  // TODO: This crashes us
  // It came from here https://stackoverflow.com/questions/18039723/c-trying-to-get-function-address-from-a-stdfunction
  typedef T(fnType)(U...);
  fnType ** fnPointer = f.template target<fnType*>();
  return (size_t) *fnPointer;
}

namespace broker {

// If the EventBroker thread calls back into subscribe or unsubscribe, this will crash, due to reacquisition of the subscriber map mutex.
class EventBroker {
public:
  using EventId = event::Event::EventId;
  using ClockType = TimerManager::ClockType;
  using TimerEndTimePoint = TimerManager::TimePoint;
  using SubscriptionId = int;
private:
  using EventHandleFunction = std::function<void(const event::Event*)>;
  struct EventSubscription {
    EventSubscription(SubscriptionId subId, EventHandleFunction &&func) : id(subId), handleFunction(func) {}
    SubscriptionId id;
    EventHandleFunction handleFunction;
    std::mutex mutex;
    int currentCallerCount{0};
    std::condition_variable conditionVariable;
  };
  using SubscriptionMapType = absl::flat_hash_map<event::EventCode, std::vector<EventSubscription*>>;
public:
  ~EventBroker();
  void runAsync();

  template<typename EventType, typename... Args>
  void publishEvent(Args&&... args) {
    const EventId eventId = getNextUniqueEventId();
    std::unique_ptr<event::Event> event = std::make_unique<EventType>(eventId, std::forward<Args>(args)...);
    VLOG(2) << absl::StreamFormat("Created event #%d, %s", event->eventId, event::toString(event->eventCode));
    publishEvent(std::move(event));
  }
  void publishEvent(event::EventCode eventCode);

  template<typename EventType, typename... Args>
  EventId publishDelayedEvent(std::chrono::milliseconds delay, Args&&... args) {
    // Abstract away TimerManager's IDs.
    const EventId eventId = getNextUniqueEventId();
    std::unique_ptr<event::Event> event = std::make_unique<EventType>(eventId, std::forward<Args>(args)...);
    VLOG(2) << absl::StreamFormat("Created delayed event #%d, %s, which triggers in %dms", event->eventId, event::toString(event->eventCode), delay.count());
    TimerManager::TimerId timerId = registerTimer(delay, std::move(event));
    addTimerIdMapping(eventId, timerId);
    return eventId;
  }

  template<typename EventType, typename... Args>
  EventId publishDelayedEvent(TimerEndTimePoint endTime, Args&&... args) {
    return publishDelayedEvent<EventType>(std::chrono::duration_cast<std::chrono::milliseconds>(endTime-ClockType::now()), std::forward<Args>(args)...);
  }
  EventId publishDelayedEvent(event::EventCode eventCode, std::chrono::milliseconds delay);
  EventId publishDelayedEvent(event::EventCode eventCode, TimerEndTimePoint endTime);

  bool cancelDelayedEvent(EventId eventId);
  std::optional<std::chrono::milliseconds> timeRemainingOnDelayedEvent(EventId eventId) const;
  std::optional<TimerEndTimePoint> delayedEventEndTime(EventId eventId) const;
  SubscriptionId subscribeToEvent(event::EventCode eventCode, EventHandleFunction &&handleFunc);

  // Will block until any events currently being sent to any subscribers are complete due to a lock on the subscriber map.
  void unsubscribeFromEvent(SubscriptionId id);
private:
  // This is used for providing unique subscription IDs. Protected by subscriptionMutex_.
  SubscriptionId subscriptionIdCounter_{0};
  // This maps events to subscriptions to these events. Protected by subscriptionMutex_.
  SubscriptionMapType subscriptionMap_;
  // This is the home of subscriptions. We require pointer stability as well as disallow copy construction/assignment. Protected by subscriptionMutex_.
  std::vector<std::unique_ptr<EventSubscription>> subscriptionMasterList_;
  std::mutex subscriptionMutex_;
  std::atomic<EventId> eventIdCounter_{0};
  TimerManager timerManager_;
  absl::flat_hash_map<EventId, TimerManager::TimerId> timerIdMap_;
  mutable std::mutex timerIdMapMutex_;

  void publishEvent(std::unique_ptr<event::Event> event);
  TimerManager::TimerId registerTimer(std::chrono::milliseconds delay, std::unique_ptr<event::Event> event);
  TimerManager::TimerId registerTimer(TimerEndTimePoint endTime, std::unique_ptr<event::Event> event);

  void notifySubscribers(std::unique_ptr<event::Event> event);
  void instantTimerFinished(event::Event *event);
  void delayedTimerFinished(event::Event *event);
  void timerFinished(event::Event *event);
  EventId getNextUniqueEventId();
  void addTimerIdMapping(EventId eventId, TimerManager::TimerId timerId);
  TimerManager::TimerId getTimerId(EventId eventId) const;
  void removeTimerIdMapping(EventId eventId);
};

} // namespace broker

#endif // BROKER_EVENT_BROKER_HPP_