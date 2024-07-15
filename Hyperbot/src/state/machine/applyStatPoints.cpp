#include "applyStatPoints.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentCharacterIncreaseIntRequest.hpp"
#include "packet/building/clientAgentCharacterIncreaseStrRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

ApplyStatPoints::ApplyStatPoints(Bot &bot, std::vector<StatPointType> statPointTypes) : StateMachine(bot), statPointTypes_(statPointTypes) {
  // stateMachineCreated(kName);
  lastInt_ = bot_.selfState().intPoints();
  lastStr_ = bot_.selfState().strPoints();
  if (bot_.selfState().getAvailableStatPoints() < statPointTypes_.size()) {
    LOG(WARNING) << "Want to apply " << statPointTypes_.size() << " stat points, but only " << bot_.selfState().getAvailableStatPoints() << " available";
    done_ = true;
    return;
  }
  LOG(INFO) << "Have " << bot_.selfState().getAvailableStatPoints() << " stat point(s). Want to apply " << statPointTypes_.size();
}

ApplyStatPoints::~ApplyStatPoints() {
  // stateMachineDestroyed();
}

void ApplyStatPoints::onUpdate(const event::Event *event) {
  if (event != nullptr) {
    if (event->eventCode == event::EventCode::kStatsChanged) {
      LOG(INFO) << "Stats changed";
      if (lastInt_ && lastStr_ &&
          bot_.selfState().intPoints() && bot_.selfState().strPoints() &&
          !statPointTypes_.empty()) {
        const auto ourType = statPointTypes_.back();
        if (ourType == StatPointType::kInt) {
          if (*bot_.selfState().intPoints() == *lastInt_ + 1) {
            LOG(INFO) << "Successfully used 1 int";
            statPointTypes_.pop_back();
            waiting_ = false;
          }
        } else {
          if (*bot_.selfState().strPoints() == *lastStr_ + 1) {
            LOG(INFO) << "Successfully used 1 str";
            statPointTypes_.pop_back();
            waiting_ = false;
          }
        }
      }
      lastInt_ = bot_.selfState().intPoints();
      lastStr_ = bot_.selfState().strPoints();
    }
  }
  
  if (waiting_) {
    return;
  }

  if (!lastInt_ || !lastStr_) {
    LOG(INFO) << "Do not yet know our current int or str";
    return;
  }

  if (statPointTypes_.empty()) {
    // Last point applied; done
    LOG(INFO) << "Last point applied. Done";
    done_ = true;
    return;
  }

  if (bot_.selfState().getAvailableStatPoints() < 1) {
    LOG(INFO) << "No more stat points";
    if (!statPointTypes_.empty()) {
      LOG(WARNING) << "Want to apply " << statPointTypes_.size() << " more stat points, but we have none left";
    }
    done_ = true;
    return;
  }

  const StatPointType pointToApply = statPointTypes_.back();
  PacketContainer packet;
  if (pointToApply == StatPointType::kInt) {
    LOG(INFO) << "Applying an Int stat point. Have " << bot_.selfState().getAvailableStatPoints() << " stat point(s)";
    packet = packet::building::ClientAgentCharacterIncreaseIntRequest::packet();
  } else {
    LOG(INFO) << "Applying a Str stat point. Have " << bot_.selfState().getAvailableStatPoints() << " stat point(s)";
    packet = packet::building::ClientAgentCharacterIncreaseStrRequest::packet();
  }
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  waiting_ = true;
}

bool ApplyStatPoints::done() const {
  return done_;
}

} // namespace state::machine