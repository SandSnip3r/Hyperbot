#include "applyStatPoints.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentCharacterIncreaseIntRequest.hpp"
#include "packet/building/clientAgentCharacterIncreaseStrRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

ApplyStatPoints::ApplyStatPoints(Bot &bot, std::vector<StatPointType> statPointTypes) : StateMachine(bot), statPointTypes_(statPointTypes) {
  // stateMachineCreated(kName);
  lastAvailableStatPoints_ = bot_.selfState().getAvailableStatPoints();
  if (lastAvailableStatPoints_ < statPointTypes_.size()) {
    LOG(WARNING) << "Want to apply " << statPointTypes_.size() << " stat points, but only " << lastAvailableStatPoints_ << " avaialable";
    done_ = true;
  }
}

ApplyStatPoints::~ApplyStatPoints() {
  // stateMachineDestroyed();
}

void ApplyStatPoints::onUpdate(const event::Event *event) {
  if (event != nullptr) {
    if (event->eventCode == event::EventCode::kCharacterAvailableStatPointsUpdated) {
      // Check if stat points decreased.
      const int difference = lastAvailableStatPoints_-bot_.selfState().getAvailableStatPoints();
      if (difference > 0) {
        LOG(INFO) << difference << " point(s) successfully applied";
        for (int i=0; i<difference; ++i) {
          statPointTypes_.pop_back();
        }
        lastAvailableStatPoints_ = bot_.selfState().getAvailableStatPoints();
        waiting_ = false;
      }
    }
  }
  
  if (waiting_) {
    return;
  }

  if (statPointTypes_.empty()) {
    // Last point applied; done
    LOG(INFO) << "Last point applied. Done";
    done_ = true;
    return;
  }

  const StatPointType pointToApply = statPointTypes_.back();
  PacketContainer packet;
  if (pointToApply == StatPointType::kInt) {
    LOG(INFO) << "Applying an Int stat point";
    packet = packet::building::ClientAgentCharacterIncreaseIntRequest::packet();
  } else {
    LOG(INFO) << "Applying a Str stat point";
    packet = packet::building::ClientAgentCharacterIncreaseStrRequest::packet();
  }
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  waiting_ = true;
}

bool ApplyStatPoints::done() const {
  return done_;
}

} // namespace state::machine