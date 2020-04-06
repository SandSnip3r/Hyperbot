#ifndef EVENT_TIMER_MANAGER_HPP_
#define EVENT_TIMER_MANAGER_HPP_

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace broker {

class TimerManager {
public:
  using TimerId = int;
  void run();
  TimerId registerTimer(std::chrono::milliseconds timerDuration, std::function<void()> timerCompletedFunction);
  void triggerInstantTimer(std::function<void()> callback);
  bool cancelTimer(TimerId id);
  ~TimerManager();
private:
  using TimePoint = std::chrono::high_resolution_clock::time_point;

  struct Timer {
    TimerId id;
    TimePoint endTime;
    std::function<void()> completionFunction;
    Timer() = default;
    Timer(TimerId tId, TimePoint et, std::function<void()> f) : id(tId), endTime(et), completionFunction(f) {}
    bool operator<(const Timer &other) const { return endTime < other.endTime; }
    bool operator>(const Timer &other) const { return endTime > other.endTime; }
  };

  bool keepRunning_{true};
  int timerIdCounter_{0};
  std::vector<Timer> timerDataHeap_;
  std::condition_variable cv_;
  std::mutex timerDataMutex_;
  std::thread thr_;
  void waitForData();
  void internalRun();
  void pruneTimers();
  void timerFinished(const Timer &timer);
};

} // namespace broker

#endif // EVENT_TIMER_MANAGER_HPP_