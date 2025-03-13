#include "resurrectInPlace.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentCharacterResurrect.hpp"

#include <absl/log/log.h>

namespace state::machine {

ResurrectInPlace::ResurrectInPlace(StateMachine *parent, bool receivedResurrectionOptionAlready) : StateMachine(parent), receivedResurrectionOptionAlready_(receivedResurrectionOptionAlready) {
  VLOG(1) << "ctor " << receivedResurrectionOptionAlready_;
}

ResurrectInPlace::~ResurrectInPlace() {
  if (requestTimeoutEventId_) {
    bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
    requestTimeoutEventId_.reset();
  }
}

Status ResurrectInPlace::onUpdate(const event::Event *event) {
  if (requestTimeoutEventId_.has_value()) {
    // We're just waiting on an event that our life state has changed.
    if (event != nullptr) {
      if (const auto *lifeStateChanged = dynamic_cast<const event::EntityLifeStateChanged*>(event)) {
        CHAR_VLOG(1) << "Someone's life state changed";
        if (lifeStateChanged->globalId == bot_.selfState()->globalId) {
          CHAR_VLOG(1) << "  it was ours " << static_cast<int>(bot_.selfState()->lifeState);
          if (bot_.selfState()->lifeState == sro::entity::LifeState::kAlive) {
            if (requestTimeoutEventId_) {
              bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
              requestTimeoutEventId_.reset();
            }
            return Status::kDone;
          }
        }
      } else if (event->eventCode == event::EventCode::kTimeout &&
                 requestTimeoutEventId_ &&
                 *requestTimeoutEventId_ == event->eventId) {
        // We timed out waiting for our life state to change.
        CHAR_VLOG(1) << "We timed out while waiting for resurrect";
        sendResurrectRequest();
      }
    }

    // TODO: Handle timeout.
    return Status::kNotDone;
  }

  if (receivedResurrectionOptionAlready_) {
    // We've already received a resurrection option; send resurrect.
    CHAR_VLOG(1) << "We've already received a resurrection option; send resurrect.";
    sendResurrectRequest();
    return Status::kNotDone;
  }

  if (event != nullptr) {
    if (const auto *resurrectOption = dynamic_cast<const event::ResurrectOption*>(event)) {
      CHAR_VLOG(1) << "Just got a resurrect option. send resurrect";
      if (resurrectOption->option != packet::enums::ResurrectionOptionFlag::kAtPresentPoint) {
        throw std::runtime_error("We can only handle resurrecting in place");
      }
      sendResurrectRequest();
      return Status::kNotDone;
    }
  }

  return Status::kNotDone;
}

void ResurrectInPlace::sendResurrectRequest() {
  // Got a resurrection option; whatever the option is, we'll do it.
  const auto packet = packet::building::ClientAgentCharacterResurrect::resurrect(packet::enums::ResurrectionOptionFlag::kAtPresentPoint);
  injectPacket(packet, PacketContainer::Direction::kBotToServer);
  requestTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(888));
}

} // namespace state::machine
