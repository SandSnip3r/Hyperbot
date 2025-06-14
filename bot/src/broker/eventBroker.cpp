#include "common/dynamicUniqueCast.hpp"
#include "eventBroker.hpp"

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>

#include <algorithm>
#include <string>

namespace broker {

EventBroker::~EventBroker() {
  VLOG(1) << "Destructing EventBroker";
}

size_t EventBroker::queuedEventCount() const {
  return timerManager_.queueSize();
}

void EventBroker::runAsync() {
  timerManager_.runAsync();
}

void EventBroker::publishEvent(event::EventCode eventCode) {
  publishEvent<event::Event>(eventCode);
}

EventBroker::EventId EventBroker::publishDelayedEvent(event::EventCode eventCode, std::chrono::milliseconds delay) {
  return publishDelayedEvent<event::Event>(delay, eventCode);
}

EventBroker::EventId EventBroker::publishDelayedEvent(event::EventCode eventCode, TimerEndTimePoint endTime) {
  return publishDelayedEvent<event::Event>(endTime, eventCode);
}

bool EventBroker::cancelDelayedEvent(EventId eventId) {
  const TimerManager::TimerId timerId = getTimerId(eventId);
  bool cancelled = timerManager_.cancelTimer(timerId);
  if (cancelled) {
    VLOG(2) << absl::StreamFormat("Cancelled event #%d", eventId);
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
  auto subscriptionIt = subscriptionMap_.find(eventCode);
  if (subscriptionIt == subscriptionMap_.end()) {
    auto itBoolResult = subscriptionMap_.emplace(eventCode, std::vector<EventSubscription*>());
    if (!itBoolResult.second) {
      throw std::runtime_error(absl::StrFormat("EventBroker::subscribeToEvent: Unable to subscribe to event %d", static_cast<int>(eventCode)));
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
  // Store the actual subscription in subscriptionMasterList_.
  subscriptionMasterList_.emplace_back(std::make_unique<EventSubscription>(thisSubscriptionId, std::move(handleFunc)));
  // Add a subscription map entry pointing to the created subscription.
  subscriptionIt->second.emplace_back(subscriptionMasterList_.back().get());
  return thisSubscriptionId;
}

void EventBroker::unsubscribeFromEvent(SubscriptionId id) {
  auto removeSubscription = [this](std::vector<EventSubscription*> &subscriptionList, std::vector<EventSubscription*>::iterator subscriptionIt, std::unique_lock<std::mutex> &thisSubscriptionLock) {
    const SubscriptionId subscriptionId = (*subscriptionIt)->id;
    // Remove the pointer to the EventSubscription object from the map.
    subscriptionList.erase(subscriptionIt);
    // Also remove the actual EventSubscription object from the list of subscriptions.
    auto it = std::find_if(subscriptionMasterList_.begin(), subscriptionMasterList_.end(), [&subscriptionId](const std::unique_ptr<EventSubscription> &subscription) {
      return subscription->id == subscriptionId;
    });
    if (it == subscriptionMasterList_.end()) {
      throw std::runtime_error("Trying to unsubscribe but this subscription does not exist in the master list");
    }
    thisSubscriptionLock.unlock();
    subscriptionMasterList_.erase(it);
  };
  std::unique_lock<std::mutex> subscriptionLock(subscriptionMutex_);
  for (SubscriptionMapType::value_type &eventSubscriptionPair : subscriptionMap_) {
    // For each event code
    std::vector<EventSubscription*> &subscriptionList = eventSubscriptionPair.second;
    for (auto subscriptionIt=subscriptionList.begin(), end=subscriptionList.end(); subscriptionIt!=end; ++subscriptionIt) {
      EventSubscription *thisSubscription = *subscriptionIt;
      if (thisSubscription->id == id) {
        // This is the one we want to unsubscribe from.
        // Another thread (it can only be the thread owned by EventBroker) might be currently calling a callback of this subscriber. To avoid issues where we could remove a subscription but still end up calling a callback for that subscription, we need to wait until all currently active callbacks of this subscription are done.
        std::unique_lock<std::mutex> thisSubscriptionLock(thisSubscription->mutex);
        if (thisSubscription->currentCallerCount == 0) {
          // Nobody is calling this subscription; done.
          removeSubscription(subscriptionList, subscriptionIt, thisSubscriptionLock);
          return;
        }
        // Someone is calling this subscription, wait until they complete and notify us.
        thisSubscription->conditionVariable.wait(thisSubscriptionLock, [&](){
          // Return true if we're done waiting.
          return thisSubscription->currentCallerCount == 0;
        });
        // Nobody is calling this subscription anymore; done.
        removeSubscription(subscriptionList, subscriptionIt, thisSubscriptionLock);
        return;
      }
    }
  }
}

void EventBroker::instantTimerFinished(std::unique_ptr<Payload> &&payload) {
  std::unique_ptr<event::Event> event = common::dynamicUniqueCast<event::Event>(std::move(payload));
  if (!event) {
    throw std::runtime_error("EventBroker::instantTimerFinished: Payload is not an event");
  }
  timerFinished(std::move(event));
}

void EventBroker::delayedTimerFinished(std::unique_ptr<Payload> &&payload) {
  std::unique_ptr<event::Event> event = common::dynamicUniqueCast<event::Event>(std::move(payload));
  if (!event) {
    throw std::runtime_error("EventBroker::instantTimerFinished: Payload is not an event");
  }
  removeTimerIdMapping(event->eventId);
  timerFinished(std::move(event));
}

void EventBroker::publishEvent(std::unique_ptr<event::Event> &&event) {
  timerManager_.triggerInstantTimer(std::move(event));
}

TimerManager::TimerId EventBroker::registerTimer(std::chrono::milliseconds delay, std::unique_ptr<event::Event> &&event) {
  // TODO: Rather than giving the event to a std::function, we should instead have an object that we give the event to which is also callable. If the object is destroyed, the event should be destroyed as well.
  return timerManager_.registerTimer(delay, std::move(event));
}

TimerManager::TimerId EventBroker::registerTimer(TimerEndTimePoint endTime, std::unique_ptr<event::Event> &&event) {
  return timerManager_.registerTimer(endTime, std::move(event));
}

void EventBroker::notifySubscribers(std::unique_ptr<event::Event> event) {
  ZoneScopedN("EventBroker::notifySubscribers");
  {
    std::string_view eventCodeString = event::toString(event->eventCode);
    ZoneName(eventCodeString.data(), eventCodeString.size());
  }
  // For each subscription pass the event to the EventHandleFunction
  std::vector<EventSubscription*> subscribersToNotify;
  {
    std::unique_lock<std::mutex> subscriptionLock(subscriptionMutex_);
    auto eventSubscriptionsIt = subscriptionMap_.find(event->eventCode);
    if (eventSubscriptionsIt != subscriptionMap_.end()) {
      std::vector<EventSubscription*> &eventSubscriptions = eventSubscriptionsIt->second;
      for (EventSubscription *eventSubscription : eventSubscriptions) {
        // Rather than call the callback while holding the subscription mutex, we will copy the subscription objects for this event, release the lock, then call them.
        std::unique_lock<std::mutex> thisSubscriptionLock(eventSubscription->mutex);
        // While we hold the lock increment the caller count.
        ++eventSubscription->currentCallerCount;
        subscribersToNotify.emplace_back(eventSubscription);
      }
    }
  }
  for (EventSubscription *eventSubscription : subscribersToNotify) {
    eventSubscription->handleFunction(event.get());
    bool shouldNotify;
    {
      std::unique_lock<std::mutex> thisSubscriptionLock(eventSubscription->mutex);
      --eventSubscription->currentCallerCount;
      shouldNotify = (eventSubscription->currentCallerCount == 0);
    }
    if (shouldNotify) {
      eventSubscription->conditionVariable.notify_one();
    }
  }
}

void EventBroker::timerFinished(std::unique_ptr<event::Event> &&event) {
  VLOG(2) << absl::StreamFormat("Event #%d %s triggered", event->eventId, event::toString(event->eventCode));
  // Take the raw pointer and move it into a unique_pointer
  notifySubscribers(std::move(event));
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
