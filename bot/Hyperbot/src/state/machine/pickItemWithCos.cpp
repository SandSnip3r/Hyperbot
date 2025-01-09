#include "pickItemWithCos.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentCosCommandRequest.hpp"

namespace state::machine {

PickItemWithCos::PickItemWithCos(Bot &bot, sro::scalar_types::EntityGlobalId cosGlobalId, sro::scalar_types::EntityGlobalId targetGlobalId) : StateMachine(bot), cosGlobalId_(cosGlobalId), targetGlobalId_(targetGlobalId) {
  stateMachineCreated(kName);
}

PickItemWithCos::~PickItemWithCos() {
  stateMachineDestroyed();
}

void PickItemWithCos::onUpdate(const event::Event *event) {
  if (event) {
    if (const auto *entityDespawnedEvent = dynamic_cast<const event::EntityDespawned*>(event)) {
      if (entityDespawnedEvent->globalId == targetGlobalId_) {
        // The item we wanted to pick up despawned
        // Whether we picked it up or not doesn't matter; we're done either way
        done_ = true;
        return;
      }
    }
  }

  if (waitingForItemToBePicked_) {
    return;
  }

  const auto packet = packet::building::ClientAgentCosCommandRequest::pickup(cosGlobalId_, targetGlobalId_);
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  waitingForItemToBePicked_ = true;
}

bool PickItemWithCos::done() const {
  return done_;
}

} // namespace state::machine