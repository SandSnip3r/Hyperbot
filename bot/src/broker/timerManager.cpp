#include "timerManager.hpp"

#include <common/TracySystem.hpp>
#include <tracy/Tracy.hpp>

#include <absl/log/log.h>

#include <algorithm>

namespace broker {

void TimerManager::runAsync() {
  if (thr_.joinable()) {
    throw std::runtime_error("TimerManager::runAsync called while already running");
  }
  thr_ = std::thread(&TimerManager::run, this);
}

bool TimerManager::cancelTimer(TimerId id) {
  bool shouldNotify=false;
  {
    std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
    if (timerDataHeap_.empty()) {
      return false;
    }
    // Save our current "next to fire" timer time
    const TimePoint prevTime = timerDataHeap_.front().endTime;
    // Remove the timer matching the id
    bool timerFound = false;
    for (auto it=timerDataHeap_.begin(), end=timerDataHeap_.end(); it!=end; ++it) {
      if (it->id == id) {
        // This is the timer that we want to cancel
        timerDataHeap_.erase(it);
        timerFound = true;
        break;
      }
    }
    if (!timerFound) {
      return false;
    }
    if (!timerDataHeap_.empty()) {
      // Reheapify if there are any timers remaining
      std::make_heap(timerDataHeap_.begin(), timerDataHeap_.end(), std::greater<Timer>());
      if (timerDataHeap_.front().endTime < prevTime) {
        // New timer ends sooner than the previous soonest
        // Wake up the thread to handle this
        shouldNotify = true;
      }
    }
  }
  if (shouldNotify) {
    cv_.notify_one();
  }
  return true;
}

std::optional<std::chrono::milliseconds> TimerManager::timeRemainingOnTimer(TimerId id) const {
  const auto currentTime = std::chrono::high_resolution_clock::now();
  std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
  for (auto it=timerDataHeap_.begin(), end=timerDataHeap_.end(); it!=end; ++it) {
    if (it->id == id) {
      return std::chrono::duration_cast<std::chrono::milliseconds>(it->endTime - currentTime);
    }
  }
  return {};
}

std::optional<TimerManager::TimePoint> TimerManager::timerEndTime(TimerId id) const {
  std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
  for (auto it=timerDataHeap_.begin(), end=timerDataHeap_.end(); it!=end; ++it) {
    if (it->id == id) {
      return it->endTime;
    }
  }
  return {};
}

TimerManager::TimerId TimerManager::registerTimer(std::chrono::milliseconds timerDuration, std::function<void()> timerCompletedFunction) {
  const auto timerEndTimePoint = std::chrono::high_resolution_clock::now() + timerDuration;
  return registerTimer(timerEndTimePoint, timerCompletedFunction);
}

TimerManager::TimerId TimerManager::registerTimer(TimePoint timeEnd, std::function<void()> timerCompletedFunction) {
  // Creating a new timer
  bool shouldNotify=false;
  TimerId thisTimerId;
  {
    std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
    TimePoint prevTime = (timerDataHeap_.empty() ? TimePoint::max() : timerDataHeap_.front().endTime);
    // Add the new timer on the "heap"
    timerDataHeap_.emplace_back(timerIdCounter_, timeEnd, timerCompletedFunction);
    std::push_heap(timerDataHeap_.begin(), timerDataHeap_.end(), std::greater<Timer>());
    if (timerDataHeap_.front().endTime < prevTime) {
      // New timer ends sooner than the previous soonest
      // Wake up the thread to handle this
      shouldNotify = true;
    }
    thisTimerId = timerIdCounter_;
    ++timerIdCounter_;
  }
  if (shouldNotify) {
    cv_.notify_one();
  }
  return thisTimerId;
}

void TimerManager::triggerInstantTimer(std::function<void()> callback) {
  {
    std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
      // Add the new "timer" on the "heap"
    timerDataHeap_.emplace_back(timerIdCounter_, std::chrono::high_resolution_clock::now(), callback);
    std::push_heap(timerDataHeap_.begin(), timerDataHeap_.end(), std::greater<Timer>());
    ++timerIdCounter_;
  }
  cv_.notify_one();
}

void TimerManager::waitForData() {
  std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
  cv_.wait(timerDataLock, [this](){ return !keepRunning_ || !timerDataHeap_.empty(); });
}

bool TimerManager::mostRecentTimerIsFinished() {
  std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
  if (!timerDataHeap_.empty()) {
    if (timerDataHeap_.front().endTime <= std::chrono::high_resolution_clock::now()) {
      return true;
    }
  }
  // Timer not finished, or no timers
  return false;
}

void TimerManager::run() {
  tracy::SetThreadName("TimerManager");
  while (keepRunning_) {
    if (timerDataHeap_.empty()) {
      waitForData();
    }

    while (!timerDataHeap_.empty()) {
      // We have data
      // Wait on shortest timer
      {
        std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
        // Double check that there is data, someone could've cancelled a timer
        if (!timerDataHeap_.empty()) {
          const auto soonestTime = timerDataHeap_.front().endTime;
          // Wait until we've reached our target time, someone has inserted a timer that will expire sooner, or there are no timers
          cv_.wait_until(timerDataLock, soonestTime, [this, &soonestTime](){
            // Double check that there is data, someone could've cancelled a timer
            if (!timerDataHeap_.empty()) {
              const auto currentSoonestTime = timerDataHeap_.front().endTime;
              // Wake up if a sooner timer has been inserted. This needs to be done because
              //  cv_.wait_until() is now going to awake at the wrong time
              const bool soonerTimerAdded = currentSoonestTime < soonestTime;
              // Wake up if our timer has expired
              const bool timeExpired = std::chrono::high_resolution_clock::now() >= currentSoonestTime;
              return (soonerTimerAdded || timeExpired);
            } else {
              // Wake up if no timer exists
              return true;
            }
          });
        }
      }
      if (mostRecentTimerIsFinished()) {
        // Woken up because our timer finished
        pruneTimers();
      }
    }
  }
}

void TimerManager::pruneTimers() {
  while (mostRecentTimerIsFinished()) {
    Timer t;
    {
      std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
      if (!timerDataHeap_.empty()) {
        t = timerDataHeap_.front();
        std::pop_heap(timerDataHeap_.begin(), timerDataHeap_.end(), std::greater<Timer>());
        timerDataHeap_.pop_back();
      }
    }
    timerFinished(t);
  }
}

void TimerManager::timerFinished(const Timer &timer) {
  timer.completionFunction();
}

TimerManager::~TimerManager() {
  VLOG(1) << "Destructing TimerManager";
  if (thr_.joinable()) {
    keepRunning_ = false;
    // Wake up the thread if it's waiting for something
    cv_.notify_one();
    // Wait for it to finish
    thr_.join();
  }
}

bool operator<(const TimerManager::Timer &lhs, const TimerManager::Timer &rhs) {
  if (lhs.endTime == rhs.endTime) {
    return lhs.id < rhs.id;
  }
  return lhs.endTime < rhs.endTime;
}

bool operator>(const TimerManager::Timer &lhs, const TimerManager::Timer &rhs) {
  if (lhs.endTime == rhs.endTime) {
    return lhs.id > rhs.id;
  }
  return lhs.endTime > rhs.endTime;
}

} // namespace broker