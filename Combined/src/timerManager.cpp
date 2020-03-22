#include "timerManager.hpp"

#include <algorithm>

namespace event {

void TimerManager::run() {
  thr_ = std::thread(&TimerManager::internalRun, this);
}

bool TimerManager::cancelTimer(TimerId id) {
  bool shouldNotify=false;
  {
    std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
    // Save our current "next to fire" timer time
    TimePoint prevTime = (timerDataHeap_.empty() ? TimePoint::max() : timerDataHeap_.front().endTime);
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
    // Reheapify
    std::make_heap(timerDataHeap_.begin(), timerDataHeap_.end(), std::greater<Timer>());
    if (timerDataHeap_.front().endTime < prevTime) {
      // New timer ends sooner than the previous soonest
      // Wake up the thread to handle this
      shouldNotify = true;
    }
  }
  if (shouldNotify) {
    cv_.notify_one();
  }
  return true;
}

TimerManager::TimerId TimerManager::registerTimer(std::chrono::milliseconds timerDuration, std::function<void()> timerCompletedFunction) {
  // Creating a new timer
  auto timerEndTimePoint = std::chrono::high_resolution_clock::now() + timerDuration;
  bool shouldNotify=false;
  {
    std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
    TimePoint prevTime = (timerDataHeap_.empty() ? TimePoint::max() : timerDataHeap_.front().endTime);
    // Add the new timer on the "heap"
    timerDataHeap_.emplace_back(timerIdCounter_, timerEndTimePoint, timerCompletedFunction);
    std::push_heap(timerDataHeap_.begin(), timerDataHeap_.end(), std::greater<Timer>());
    if (timerDataHeap_.front().endTime < prevTime) {
      // New timer ends sooner than the previous soonest
      // Wake up the thread to handle this
      shouldNotify = true;
    }
  }
  if (shouldNotify) {
    cv_.notify_one();
  }
  const auto thisTimerId = timerIdCounter_;
  ++timerIdCounter_;
  return thisTimerId;
}

void TimerManager::waitForData() {
  std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
  cv_.wait(timerDataLock, [this](){ return !keepRunning_ || !timerDataHeap_.empty(); });
}

void TimerManager::internalRun() {
  while (keepRunning_) {
    if (timerDataHeap_.empty()) {
      waitForData();
    }

    while (!timerDataHeap_.empty()) {
      // We have data
      // Wait on shortest timer
      {
        std::unique_lock<std::mutex> timerDataLock(timerDataMutex_);
        const auto soonestTime = timerDataHeap_.front().endTime;
        // Wait until we've reached our target time or someone has inserted a timer that will expire sooner
        cv_.wait_until(timerDataLock, soonestTime, [this, &soonestTime](){
          const auto currentSoonestTime = timerDataHeap_.front().endTime;
          // Wake up if a sooner timer has been inserted. This needs to be done because
          //  cv_.wait_until() is now going to awake at the wrong time
          const bool soonerTimerAdded = currentSoonestTime < soonestTime;
          // Wake up if our timer has expired
          const bool timeExpired = std::chrono::high_resolution_clock::now() >= currentSoonestTime;
          return (soonerTimerAdded || timeExpired);
        });
      }
      if (std::chrono::high_resolution_clock::now() >= timerDataHeap_.front().endTime) {
        // Woken up because our timer finished
        pruneTimers();
      }
    }
  }
}

void TimerManager::pruneTimers() {
  while (!timerDataHeap_.empty() && timerDataHeap_.front().endTime <= std::chrono::high_resolution_clock::now()) {
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
  keepRunning_ = false;
  // Wake up the thread if it's waiting for something
  cv_.notify_one();
  // Wait for it to finish
  thr_.join();
}

} // namespace event