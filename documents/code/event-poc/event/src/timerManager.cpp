#include "../include/timerManager.hpp"

namespace event {

void TimerManager::run() {
  thr = std::thread(&TimerManager::internalRun, this);
}

void TimerManager::registerTimer(std::chrono::milliseconds timerDuration, std::function<void()> timerCompletedFunction) {
  // Creating a new timer
  auto timerEndTimePoint = std::chrono::high_resolution_clock::now() + timerDuration;
  TimePoint prevTime;
  bool shouldNotify=false;
  {
    std::unique_lock<std::mutex> timerDataLock(timerDataMutex);
    prevTime = (timerData.empty() ? TimePoint::max() : timerData.top().endTime);
    timerData.emplace(timerEndTimePoint, timerCompletedFunction);
    if (timerData.top().endTime < prevTime) {
      // New timer ends sooner than the previous soonest
      // Wake up the thread to handle this
      shouldNotify = true;
    }
  }
  if (shouldNotify) {
    cv.notify_one();
  }
}

void TimerManager::waitForData() {
  std::unique_lock<std::mutex> timerDataLock(timerDataMutex);
  cv.wait(timerDataLock, [this](){ return !timerData.empty(); });
}

void TimerManager::internalRun() {
  while (true) {
    if (timerData.empty()) {
      waitForData();
    }

    while (!timerData.empty()) {
      // We have data
      // Wait on shortest timer
      {
        std::unique_lock<std::mutex> timerDataLock(timerDataMutex);
        // Wait until we've reached our target time or someone has inserted a timer that will expire sooner
        cv.wait_until(timerDataLock, timerData.top().endTime, [this](){ return std::chrono::high_resolution_clock::now() >= timerData.top().endTime; });
      }
      if (std::chrono::high_resolution_clock::now() >= timerData.top().endTime) {
        // Woken up because our timer finished
        pruneTimers();
      }
    }
  }
}

void TimerManager::pruneTimers() {
  while (!timerData.empty() && timerData.top().endTime <= std::chrono::high_resolution_clock::now()) {
    Timer t;
    {
      std::unique_lock<std::mutex> timerDataLock(timerDataMutex);
      if (!timerData.empty()) {
        t = timerData.top();
        timerData.pop();
      }
    }
    timerFinished(t);
  }
}

void TimerManager::timerFinished(const Timer &timer) {
  timer.completionFunction();
}

TimerManager::~TimerManager() {
  thr.join();
}

} // namespace event