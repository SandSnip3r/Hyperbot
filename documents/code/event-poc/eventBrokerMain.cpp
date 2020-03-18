#include "event/include/event.hpp"
#include "event/include/eventBroker.hpp"

#include <iostream>
#include <unistd.h>

using namespace std;

void handleEventFunc(const std::unique_ptr<event::Event> &event) {
  std::cout << "Event received! " << static_cast<int>(event->getEventCode()) << '\n';
}

// enum class EventCode {
//   kTest1,
//   kTest2
// };

int main() {
  event::EventBroker eventBroker;
  eventBroker.run();
  eventBroker.subscribeToEvent(event::EventCode::kTest1, handleEventFunc);
  eventBroker.subscribeToEvent(event::EventCode::kTest2, handleEventFunc);

  cout << "Publishing 1 now\n";
  eventBroker.publishEvent(std::make_unique<event::Event>(event::EventCode::kTest1));
  cout << "Publishing 2 now\n";
  eventBroker.publishEvent(std::make_unique<event::Event>(event::EventCode::kTest2));
  
  cout << "Publishing 1 with 1 second delay\n";
  eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTest1), std::chrono::milliseconds(1000));
  cout << "Publishing 2 with 1 second delay\n";
  eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTest2), std::chrono::milliseconds(1000));
  
  cout << "Publishing 1 with 7 second delay\n";
  eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTest1), std::chrono::milliseconds(7000));
  cout << "Publishing 2 with 8 second delay\n";
  eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTest2), std::chrono::milliseconds(8000));

  usleep(10000000);

//   void run();
//   bool publishEvent(std::unique_ptr<Event> event);
  
//   //TODO: return timer Id
//   bool publishDelayedEvent(std::unique_ptr<Event> event, std::chrono::milliseconds delay);

//   //TODO: return subscription Id
//   void subscribeToEvent(EventCode eventCode, EventHandleFunction &&handleFunc);
// private:
//   TimerManager timerManager_;
//   void timerFinished(Event *event);
  return 0;
}