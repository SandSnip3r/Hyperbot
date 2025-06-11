#ifndef BROKER_TIMER_MANAGER_HPP_
#define BROKER_TIMER_MANAGER_HPP_

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace broker {

// This class holds the data that the TimerManager will pass to the callback target when a timer finishes.
class Payload {
public:
  virtual ~Payload() = 0;
};

// This should be derived from by the class that will receive the timer-finished callbacks.
class CallbackTarget {
public:
  virtual void instantTimerFinished(std::unique_ptr<Payload> &&payload) = 0;
  virtual void delayedTimerFinished(std::unique_ptr<Payload> &&payload) = 0;
};

class TimerManager {
public:
  using TimerId = uint32_t;
  using ClockType = std::chrono::steady_clock;
  using TimePoint = ClockType::time_point;

  TimerManager(CallbackTarget &callbackTarget);
  ~TimerManager();

  void runAsync();
  TimerId registerTimer(std::chrono::milliseconds timerDuration, std::unique_ptr<Payload> &&payload);
  TimerId registerTimer(TimePoint timeEnd, std::unique_ptr<Payload> &&payload);
  void triggerInstantTimer(std::unique_ptr<Payload> &&payload);
  bool cancelTimer(TimerId id);
  std::optional<std::chrono::milliseconds> timeRemainingOnTimer(TimerId id) const;
  std::optional<TimePoint> timerEndTime(TimerId id) const;

  /// Returns the number of pending timers.
  size_t queueSize() const;

private:
  struct Timer {
    TimerId id;
    bool isInstant;
    TimePoint endTime;
    std::unique_ptr<Payload> payload;
    Timer() = default;
    Timer(TimerId tId, bool isInstant, TimePoint et, std::unique_ptr<Payload> &&p) : id(tId), isInstant(isInstant), endTime(et), payload(std::move(p)) {}
    Timer& operator=(Timer&& other) noexcept {
      if (this != &other) {
        id = other.id;
        isInstant = other.isInstant;
        endTime = other.endTime;
        payload = std::move(other.payload); // Transfer ownership of unique_ptr
      }
      return *this;
    }
    Timer(Timer&& other) noexcept {
      id = other.id;
      isInstant = other.isInstant;
      endTime = other.endTime;
      payload = std::move(other.payload); // Transfer ownership of unique_ptr
    }
  };
  friend bool operator<(const Timer &lhs, const Timer &rhs);
  friend bool operator>(const Timer &lhs, const Timer &rhs);

  CallbackTarget &callbackTarget_;
  bool keepRunning_{true};
  TimerId timerIdCounter_{0};
  std::vector<Timer> timerDataHeap_;
  std::condition_variable cv_;
  mutable std::mutex timerDataMutex_;
  std::thread thr_;
  void waitForData();
  void run();
  void pruneTimers();
  void timerFinished(Timer &&timer);
  bool mostRecentTimerIsFinished();
};


bool operator<(const TimerManager::Timer &lhs, const TimerManager::Timer &rhs);
bool operator>(const TimerManager::Timer &lhs, const TimerManager::Timer &rhs);

} // namespace broker

#endif // BROKER_TIMER_MANAGER_HPP_