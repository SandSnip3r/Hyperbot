#ifndef EVENT_TIMER_MANAGER_HPP_
#define EVENT_TIMER_MANAGER_HPP_

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace broker {

class TimerManager {
public:
  using TimerId = uint32_t;
  using ClockType = std::chrono::high_resolution_clock;
  using TimePoint = ClockType::time_point;

  void runAsync();
  TimerId registerTimer(std::chrono::milliseconds timerDuration, std::function<void()> timerCompletedFunction);
  TimerId registerTimer(TimePoint timeEnd, std::function<void()> timerCompletedFunction);
  void triggerInstantTimer(std::function<void()> callback);
  bool cancelTimer(TimerId id);
  std::optional<std::chrono::milliseconds> timeRemainingOnTimer(TimerId id) const;
  std::optional<TimePoint> timerEndTime(TimerId id) const;

  ~TimerManager();

private:
  struct Timer {
    TimerId id;
    TimePoint endTime;
    std::function<void()> completionFunction;
    Timer() = default;
    Timer(TimerId tId, TimePoint et, std::function<void()> f) : id(tId), endTime(et), completionFunction(f) {}
  };
  friend bool operator<(const Timer &lhs, const Timer &rhs);
  friend bool operator>(const Timer &lhs, const Timer &rhs);

  bool keepRunning_{true};
  TimerId timerIdCounter_{0};
  std::vector<Timer> timerDataHeap_;
  std::condition_variable cv_;
  mutable std::mutex timerDataMutex_;
  std::thread thr_;
  void waitForData();
  void run();
  void pruneTimers();
  void timerFinished(const Timer &timer);
  bool mostRecentTimerIsFinished();
};


bool operator<(const TimerManager::Timer &lhs, const TimerManager::Timer &rhs);
bool operator>(const TimerManager::Timer &lhs, const TimerManager::Timer &rhs);

} // namespace broker

#endif // EVENT_TIMER_MANAGER_HPP_