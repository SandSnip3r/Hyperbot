#include "eventBroker.hpp"
#include "logging.hpp"

#include <string>

namespace broker {

void EventBroker::runAsync() {
  timerManager_.runAsync();
}

void EventBroker::publishEvent(std::unique_ptr<event::Event> event) {
  timerManager_.triggerInstantTimer(std::bind(&EventBroker::timerFinished, this, event.release()));
}

EventBroker::DelayedEventId EventBroker::publishDelayedEvent(std::unique_ptr<event::Event> event, std::chrono::milliseconds delay) {
  return timerManager_.registerTimer(delay, std::bind(&EventBroker::timerFinished, this, event.release()));
}

bool EventBroker::cancelDelayedEvent(DelayedEventId id) {
  return timerManager_.cancelTimer(id);
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
    //     std::cout << "This exact handler is even subscribed to this event" << std::endl;
    //   }
    // }
  }
  const auto thisSubscriptionId = subscriptionIdCounter_;
  subscriptionIt->second.emplace_back(subscriptionIdCounter_, std::move(handleFunc));
  ++subscriptionIdCounter_;
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

void EventBroker::timerFinished(event::Event *event) {
  // Take the raw pointer and move it into a unique_pointer
  notifySubscribers(std::unique_ptr<event::Event>(event));
}

} // namespace broker
