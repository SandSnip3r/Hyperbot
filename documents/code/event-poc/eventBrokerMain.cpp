#include "event/include/event.hpp"
#include "event/include/eventBroker.hpp"

#include <array>
#include <iostream>
#include <map>
#include <algorithm>
#include <random>
#include <unistd.h>

using namespace std;

mt19937 createRandomEngine() {
  random_device rd;
  const auto kSeed = rd();
  cout << "Seed: " << kSeed << '\n';
  // array<int, mt19937::state_size> seed_data;
  // generate_n(seed_data.data(), seed_data.size(), ref(rd));
  // seed_seq seq(begin(seed_data), end(seed_data));
  // return mt19937(seq);
  return mt19937(kSeed);
}

class SpecificEvent : public event::Event {
public:
  explicit SpecificEvent(event::EventCode eventCode, int id) : Event(eventCode), id_(id) {}
  int getId() const { return id_; }
  virtual ~SpecificEvent() = default;
private:
  int id_;
};

class EventBrokerTester {
public:
  EventBrokerTester() {
    // Start the eventBroker
    eventBroker_.run();
  }
  void run() {
    // First, subscribe to events
    eventBroker_.subscribeToEvent(event::EventCode::kTest1, std::bind(&EventBrokerTester::handleEventFunc, this, placeholders::_1));
    auto eng = createRandomEngine();
    const int kEventInsertionCount=100000;
    const int kMaxDelayMs=20000;
    int id=0;
    bernoulli_distribution createDist(0.9);
    bernoulli_distribution delayedDist(0.9);
    uniform_int_distribution<int> delayDist(0, kMaxDelayMs);
    vector<event::EventBroker::DelayedEventId> delayedEvents;
    delayedEvents.reserve(kEventInsertionCount*0.90);
    for (int i=0; i<kEventInsertionCount; ++i) {
      if ((i+1) % 100000 == 0) {
        cout << "Registering #" << i << '\n';
      }
      if (createDist(eng)) {
        // Create an event
        if (delayedDist(eng)) {
          // Trigger a delayed event
          const auto kDelay = delayDist(eng);
          const auto kDelayMs = chrono::milliseconds(kDelay);
          const auto currTime = currentTime();
          sentEvents_.emplace(id, currTime+kDelayMs);
          auto delayedEventId = eventBroker_.publishDelayedEvent(unique_ptr<event::Event>(new SpecificEvent(event::EventCode::kTest1, id)), kDelayMs);
          delayedEvents.push_back(delayedEventId);
        } else {
          // Trigger an instant event
          sentEvents_.emplace(id, currentTime());
          eventBroker_.publishEvent(unique_ptr<event::Event>(new SpecificEvent(event::EventCode::kTest1, id)));
        }
        ++id;
      } else {
        // Delete a delayed event
        uniform_int_distribution<int> deleteWhichDist(0, delayedEvents.size()-1);
        const auto whichToDelete = deleteWhichDist(eng);
        const auto it = delayedEvents.begin()+whichToDelete;
        bool cancelled = eventBroker_.cancelDelayedEvent(*it);
        if (cancelled) {
          cout << "Cancelled " << *it << "\n";
        } else {
          cout << "Whoops\n";
        }
        delayedEvents.erase(it);
      }
    }

    // Wait for all events to be triggered
    cout << "Waiting " << kMaxDelayMs << "ms for all events\n";
    usleep(kMaxDelayMs*1000);

    validateResults();
    for (int i=0; i<1000; ++i) {
      cout << "DONE!\n";
    }
  }
private:
  using TimePoint = std::chrono::high_resolution_clock::time_point;
  event::EventBroker eventBroker_;
  map<int, TimePoint> receivedEvents_;
  map<int, TimePoint> sentEvents_;

  int64_t toMsSinceEpoch(TimePoint tp) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
  }

  void validateResults() {
    // For every event in sentEvents_ it should exist in receivedEvents_ with roughly the same timestamp
    for (const auto &idTimePair : sentEvents_) {
      const auto originalTp = toMsSinceEpoch(idTimePair.second);
      cout << idTimePair.first << ' ';
      auto it = receivedEvents_.find(idTimePair.first);
      if (it == receivedEvents_.end()) {
        cout << "Never Received!!!!\n";
      } else {
        cout << toMsSinceEpoch(it->second) - originalTp << '\n';
      }
    }
    cout << "Validation complete\n";
  }

  TimePoint currentTime() const {
    return std::chrono::high_resolution_clock::now();
  }

  void handleEventFunc(const std::unique_ptr<event::Event> &event) {
    const auto *eventPtr = dynamic_cast<const SpecificEvent*>(event.get());
    if (eventPtr == nullptr) {
      cerr << "Wait, what\n";
      return;
    }
    const auto eventId = eventPtr->getId();
    const auto currTime = currentTime();
    receivedEvents_.emplace(eventId, currTime);
  }
};

int main() {
  EventBrokerTester tester;
  tester.run();
  return 0;
}

/* void handleEventFunc(const std::unique_ptr<event::Event> &event) {
  std::cout << "handleEventFunc: Event received! " << static_cast<int>(event->getEventCode()) << '\n';
}

int main() {
  event::EventBroker eventBroker;
  eventBroker.run();
  auto sub1Id = eventBroker.subscribeToEvent(event::EventCode::kTest1, handleEventFunc);
  // cout << "sub1Id: " << sub1Id << '\n';
  auto sub2Id = eventBroker.subscribeToEvent(event::EventCode::kTest2, handleEventFunc);
  // cout << "sub2Id: " << sub2Id << '\n';

  // cout << "Publishing 1 now\n";
  eventBroker.publishEvent(std::make_unique<event::Event>(event::EventCode::kTest1));
  // cout << "Publishing 2 now\n";
  eventBroker.publishEvent(std::make_unique<event::Event>(event::EventCode::kTest2));
  
  // cout << "Publishing 1 with 1 second delay, ";
  auto delayedEventId0 = eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTest1), std::chrono::milliseconds(1000));
  // cout << "id: " << delayedEventId0 << '\n';

  // cout << "Publishing 2 with 1 second delay, ";
  auto delayedEventId1 = eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTest2), std::chrono::milliseconds(1000));
  // cout << "id: " << delayedEventId1 << '\n';

  
  // cout << "Publishing 1 with 8 second delay, ";
  auto delayedEventId2 = eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTest1), std::chrono::milliseconds(8000));
  // cout << "id: " << delayedEventId2 << '\n';

  // cout << "Publishing 2 with 8 second delay, ";
  auto delayedEventId3 = eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTest2), std::chrono::milliseconds(8000));
  // cout << "id: " << delayedEventId3 << '\n';

  std::cout << "Sleeping 5s\n";
  usleep(5000000);

  // Unsubscribe from one
  // cout << "Unsubscribing from " << sub1Id << "\n";
  eventBroker.unsubscribeFromEvent(sub1Id);

  // Cancel another
  // cout << "Canceling " << delayedEventId3 << "\n";
  eventBroker.cancelDelayedEvent(delayedEventId3);
  eventBroker.cancelDelayedEvent(delayedEventId3);
  eventBroker.cancelDelayedEvent(delayedEventId3);

  std::cout << "Sleeping 5s\n";
  usleep(5000000);
  eventBroker.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTest1), std::chrono::milliseconds(1000));
  eventBroker.subscribeToEvent(event::EventCode::kTest1, handleEventFunc);
  
  std::cout << "Sleeping 5s\n";
  usleep(5000000);
  
  return 0;
} */