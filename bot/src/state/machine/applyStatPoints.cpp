#include "applyStatPoints.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentCharacterIncreaseIntRequest.hpp"
#include "packet/building/clientAgentCharacterIncreaseStrRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

ApplyStatPoints::ApplyStatPoints(Bot &bot, std::vector<StatPointType> statPointTypes) : StateMachine(bot), statPointTypes_(statPointTypes) {}
ApplyStatPoints::ApplyStatPoints(StateMachine *parent, std::vector<StatPointType> statPointTypes) : StateMachine(parent), statPointTypes_(statPointTypes) {}

ApplyStatPoints::~ApplyStatPoints() {}

Status ApplyStatPoints::onUpdate(const event::Event *event) {
  if (!initialized_) {
    const Status status = initialize();
    initialized_ = true;
    if (status == Status::kDone) {
      return Status::kDone;
    }
  }

  if (event != nullptr) {
    if (event->eventCode == event::EventCode::kStatsChanged) {
      VLOG(3) << "Stats changed";
      if (lastInt_ && lastStr_ &&
          bot_.selfState()->intPoints() && bot_.selfState()->strPoints() &&
          !statPointTypes_.empty()) {
        const auto ourType = statPointTypes_.back();
        if (ourType == StatPointType::kInt) {
          if (*bot_.selfState()->intPoints() == *lastInt_ + 1) {
            VLOG(2) << "Successfully used 1 int";
            statPointTypes_.pop_back();
            success();
          }
        } else {
          if (*bot_.selfState()->strPoints() == *lastStr_ + 1) {
            VLOG(2) << "Successfully used 1 str";
            statPointTypes_.pop_back();
            success();
          }
        }
      }
      lastInt_ = bot_.selfState()->intPoints();
      lastStr_ = bot_.selfState()->strPoints();
    } else if (event->eventCode == event::EventCode::kTimeout) {
      if (timeoutEventId_ && event->eventId == *timeoutEventId_) {
        VLOG(3) << "Timed out!";
        timeoutEventId_.reset();
      }
    }
  }

  if (timeoutEventId_) {
    return Status::kNotDone;
  }

  if (!lastInt_ || !lastStr_) {
    VLOG(2) << "Do not yet know our current int or str";
    return Status::kNotDone;
  }

  if (statPointTypes_.empty()) {
    // Last point applied; done
    VLOG(3) << "Last point applied. Done";
    return Status::kDone;
  }

  if (bot_.selfState()->getAvailableStatPoints() < 1) {
    VLOG(2) << "No more stat points";
    if (!statPointTypes_.empty()) {
      LOG(WARNING) << "Want to apply " << statPointTypes_.size() << " more stat points, but we have none left";
    }
    return Status::kDone;
  }

  const StatPointType pointToApply = statPointTypes_.back();
  PacketContainer packet;
  if (pointToApply == StatPointType::kInt) {
    VLOG(2) << "Applying an Int stat point. Have " << bot_.selfState()->getAvailableStatPoints() << " stat point(s)";
    packet = packet::building::ClientAgentCharacterIncreaseIntRequest::packet();
  } else {
    VLOG(2) << "Applying a Str stat point. Have " << bot_.selfState()->getAvailableStatPoints() << " stat point(s)";
    packet = packet::building::ClientAgentCharacterIncreaseStrRequest::packet();
  }
  injectPacket(packet, PacketContainer::Direction::kBotToServer);
  timeoutEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(200));
  return Status::kNotDone;
}

void ApplyStatPoints::success() {
  if (!timeoutEventId_) {
    LOG(WARNING) << "Successfully added stat point, but had no timeout event";
    return;
  }
  bool success = bot_.eventBroker().cancelDelayedEvent(*timeoutEventId_);
  if (!success) {
    throw std::runtime_error("Failed to cancel timer for stat point application timeout");
  }
  timeoutEventId_.reset();
}

Status ApplyStatPoints::initialize() {
  lastInt_ = bot_.selfState()->intPoints();
  lastStr_ = bot_.selfState()->strPoints();
  if (bot_.selfState()->getAvailableStatPoints() < statPointTypes_.size()) {
    LOG(WARNING) << "Want to apply " << statPointTypes_.size() << " stat points, but only " << bot_.selfState()->getAvailableStatPoints() << " available";
    return Status::kDone;
  }
  // Apply the given stat points in the order given (first to last), but since we want to be efficient with our vector, we'll remove items from the end. Because of this, we reverse the vector.
  std::reverse(statPointTypes_.begin(), statPointTypes_.end());
  VLOG(1) << "Have " << bot_.selfState()->getAvailableStatPoints() << " stat point(s). Want to apply " << statPointTypes_.size();
  return Status::kNotDone;
}

} // namespace state::machine