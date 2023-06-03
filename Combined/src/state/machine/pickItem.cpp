#include "pickItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"

namespace state::machine {

PickItem::PickItem(Bot &bot, sro::scalar_types::EntityGlobalId targetGlobalId) : StateMachine(bot), targetGlobalId_(targetGlobalId) {
  stateMachineCreated(kName);
}

PickItem::~PickItem() {
  stateMachineDestroyed();
}

void PickItem::onUpdate(const event::Event *event) {
  // At this point, we are within range of the item and can pick it up
  if (event) {
    if (const auto *entityDespawnedEvent = dynamic_cast<const event::EntityDespawned*>(event)) {
      if (entityDespawnedEvent->globalId == targetGlobalId_) {
        // The item we wanted to pick up despawned
        // Whether we picked it up or not doesn't matter; we're done either way
        done_ = true;
        return;
      }
    }
    // TODO: Handle response for CommandRequest
  }

  if (waitingForItemToBePicked_) {
    return;
  }

  const auto packet = packet::building::ClientAgentActionCommandRequest::pickup(targetGlobalId_);
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  waitingForItemToBePicked_ = true;
}

bool PickItem::done() const {
  return done_;
}

} // namespace state::machine