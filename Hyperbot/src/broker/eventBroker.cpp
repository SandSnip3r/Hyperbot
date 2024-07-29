#include "eventBroker.hpp"

#include <absl/log/log.h>

#include <string>

namespace broker {

void EventBroker::runAsync() {
  timerManager_.runAsync();
}

void EventBroker::publishEvent(event::EventCode eventCode) {
  publishEvent<event::Event>(eventCode);
}

EventBroker::EventId EventBroker::publishDelayedEvent(std::chrono::milliseconds delay, event::EventCode eventCode) {
  return publishDelayedEvent<event::Event>(delay, eventCode);
}

EventBroker::EventId EventBroker::publishDelayedEvent(TimerEndTimePoint endTime, event::EventCode eventCode) {
  return publishDelayedEvent<event::Event>(endTime, eventCode);
}

bool EventBroker::cancelDelayedEvent(EventId eventId) {
  const TimerManager::TimerId timerId = getTimerId(eventId);
  bool cancelled = timerManager_.cancelTimer(timerId);
  if (cancelled) {
    removeTimerIdMapping(eventId);
  } else {
    LOG(WARNING) << "Tried to cancel delayed event but failed";
  }
  return cancelled;
}

std::optional<std::chrono::milliseconds> EventBroker::timeRemainingOnDelayedEvent(EventId eventId) const {
  const TimerManager::TimerId timerId = getTimerId(eventId);
  return timerManager_.timeRemainingOnTimer(timerId);
}

std::optional<EventBroker::TimerEndTimePoint> EventBroker::delayedEventEndTime(EventId eventId) const {
  const TimerManager::TimerId timerId = getTimerId(eventId);
  return timerManager_.timerEndTime(timerId);
}

EventBroker::SubscriptionId EventBroker::subscribeToEvent(event::EventCode eventCode, EventHandleFunction &&handleFunc) {
  std::unique_lock<std::mutex> subscriptionLock(subscriptionMutex_);
  auto subscriptionIt = subscriptions_.find(eventCode);
  if (subscriptionIt == subscriptions_.end()) {
    auto itBoolResult = subscriptions_.emplace(eventCode, std::vector<EventSubscription>());
    if (!itBoolResult.second) {
      throw std::runtime_error("EventBroker::subscribeToEvent: Unable to subscribe to event "+std::to_string(static_cast<int>(eventCode)));
    } else {
      subscriptionIt = itBoolResult.first;
    }
  } else {
    // Already subscribed to this event
    // TODO: Check if this handler is one of the ones already subscribed
    // const auto thisAddr = getAddress(handleFunc);
    // for (const auto &i : subscriptionIt->second) {
    //   const auto iAddr = getAddress(i.handleFunction);
    //   if (thisAddr == iAddr) {
    //     LOG(WARNING) << "This exact handler is even subscribed to this event";
    //   }
    // }
  }

  const SubscriptionId thisSubscriptionId = subscriptionIdCounter_++;
  subscriptionIt->second.emplace_back(thisSubscriptionId, std::move(handleFunc));
  return thisSubscriptionId;
}

void EventBroker::unsubscribeFromEvent(SubscriptionId id) {
  std::unique_lock<std::mutex> subscriptionLock(subscriptionMutex_);
  for (auto &eventSubscriptionPair : subscriptions_) {
    // For each event code
    auto &subscriptionList = eventSubscriptionPair.second;
    for (auto subscriptionIt=subscriptionList.begin(), end=subscriptionList.end(); subscriptionIt!=end; ++subscriptionIt) {
      if (subscriptionIt->id == id) {
        // This is the one we want to unsubscribe from
        subscriptionList.erase(subscriptionIt);
        return;
      }
    }
  }
}

void EventBroker::publishEvent(std::unique_ptr<event::Event> event) {
  timerManager_.triggerInstantTimer(std::bind(&EventBroker::instantTimerFinished, this, event.release()));
}

TimerManager::TimerId EventBroker::registerTimer(std::chrono::milliseconds delay, std::unique_ptr<event::Event> event) {
  return timerManager_.registerTimer(delay, std::bind(&EventBroker::delayedTimerFinished, this, event.release()));
}

TimerManager::TimerId EventBroker::registerTimer(TimerEndTimePoint endTime, std::unique_ptr<event::Event> event) {
  return timerManager_.registerTimer(endTime, std::bind(&EventBroker::delayedTimerFinished, this, event.release()));
}

void EventBroker::notifySubscribers(std::unique_ptr<event::Event> event) {
  // For each subscription pass the event to the EventHandleFunction
  std::unique_lock<std::mutex> subscriptionLock(subscriptionMutex_);
  auto eventSubscriptionsIt = subscriptions_.find(event->eventCode);
  if (eventSubscriptionsIt != subscriptions_.end()) {
    auto &eventSubscriptions = eventSubscriptionsIt->second;
    for (auto &eventSubscription : eventSubscriptions) {
      eventSubscription.handleFunction(event.get());
    }
  }
}

void EventBroker::instantTimerFinished(event::Event *event) {
  timerFinished(event);
}

void EventBroker::delayedTimerFinished(event::Event *event) {
  removeTimerIdMapping(event->eventId);
  timerFinished(event);
}

void EventBroker::timerFinished(event::Event *event) {
  VLOG(1) << "Publishing event " << event::toString(event->eventCode);
  // Take the raw pointer and move it into a unique_pointer
  notifySubscribers(std::unique_ptr<event::Event>(event));
}

event::Event::EventId EventBroker::getNextUniqueEventId() {
  return eventIdCounter_++;
}

void EventBroker::addTimerIdMapping(EventId eventId, TimerManager::TimerId timerId) {
  std::unique_lock<std::mutex> lock(timerIdMapMutex_);
  timerIdMap_[eventId] = timerId;
}

TimerManager::TimerId EventBroker::getTimerId(EventId eventId) const {
  std::unique_lock<std::mutex> lock(timerIdMapMutex_);
  auto it = timerIdMap_.find(eventId);
  if (it == timerIdMap_.end()) {
    throw std::runtime_error("getTimerId: Have no timer id for event "+std::to_string(eventId));
  }
  return it->second;
}

void EventBroker::removeTimerIdMapping(EventId eventId) {
  std::unique_lock<std::mutex> lock(timerIdMapMutex_);
  auto it = timerIdMap_.find(eventId);
  if (it == timerIdMap_.end()) {
    throw std::runtime_error("removeTimerIdMapping: Have no timer id for event "+std::to_string(eventId));
  }
  timerIdMap_.erase(it);
}

} // namespace broker
