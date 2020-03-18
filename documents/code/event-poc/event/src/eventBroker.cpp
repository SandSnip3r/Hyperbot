#include "../include/eventBroker.hpp"

#include <iostream>

namespace event {

void EventBroker::run() {
  timerManager_.run();
}

bool EventBroker::publishEvent(std::unique_ptr<Event> event) {
  // For each subscription pass the event to the EventHandleFunction
  auto handlersIt = subscriptions_.find(event->getEventCode());
  if (handlersIt != subscriptions_.end()) {
    auto &handlers = handlersIt->second;
    for (auto &handler : handlers) {
      handler(event);
    }
  }
}

bool EventBroker::publishDelayedEvent(std::unique_ptr<Event> event, std::chrono::milliseconds delay) {
  timerManager_.registerTimer(delay, std::bind(&EventBroker::timerFinished, this, event.release()));
}

void EventBroker::subscribeToEvent(EventCode eventCode, EventHandleFunction &&handleFunc) {
  auto subscriptionIt = subscriptions_.find(eventCode);
  if (subscriptionIt == subscriptions_.end()) {
    auto itBoolResult = subscriptions_.emplace(eventCode, std::vector<EventHandleFunction>());
    if (!itBoolResult.second) {
      std::cerr << "Unable to subscribe!\n";
      // TODO: Handle error better
      return;
    } else {
      subscriptionIt = itBoolResult.first;
    }
  }
  subscriptionIt->second.emplace_back(std::move(handleFunc));
}

void EventBroker::timerFinished(Event *event) {
  // Take the raw pointer and move it into a unique_pointer
  publishEvent(std::unique_ptr<Event>(event));
}

} // namespace event
