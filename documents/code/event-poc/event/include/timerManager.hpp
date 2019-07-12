#ifndef TIMER_MANAGER_HPP
#define TIMER_MANAGER_HPP

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace event {

class TimerManager {
public:
  void run();

  // TODO: return timer Id
  void registerTimer(std::chrono::milliseconds timerDuration, std::function<void()> timerCompletedFunction);
  ~TimerManager();
private:
  using TimePoint = std::chrono::high_resolution_clock::time_point;

  struct Timer {
    TimePoint endTime;
    std::function<void()> completionFunction;
    Timer() = default;
    Timer(TimePoint et, std::function<void()> f) : endTime(et), completionFunction(f) {}
    bool operator<(const Timer &other) const { return endTime < other.endTime; }
    bool operator>(const Timer &other) const { return endTime > other.endTime; }
  };

  std::priority_queue<Timer, std::vector<Timer>, std::greater<Timer>> timerData;
  std::condition_variable cv;
  std::mutex timerDataMutex;
  std::thread thr;
  void waitForData();
  void internalRun();
  void pruneTimers();
  void timerFinished(const Timer &timer);
};

} // namespace event

#endif // TIMER_MANAGER_HPP