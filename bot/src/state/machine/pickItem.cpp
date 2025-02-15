#include "pickItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"

namespace state::machine {

PickItem::PickItem(Bot &bot, sro::scalar_types::EntityGlobalId targetGlobalId) : StateMachine(bot), targetGlobalId_(targetGlobalId) {
  stateMachineCreated(kName);
}

PickItem::~PickItem() {
  stateMachineDestroyed();
}

Status PickItem::onUpdate(const event::Event *event) {
  // At this point, we are within range of the item and can pick it up
  if (event) {
    if (const auto *entityDespawnedEvent = dynamic_cast<const event::EntityDespawned*>(event)) {
      if (entityDespawnedEvent->globalId == targetGlobalId_) {
        // The item we wanted to pick up despawned
        // Whether we picked it up or not doesn't matter; we're done either way
        return Status::kDone;
      }
    }
    // TODO: Handle response for CommandRequest
  }

  if (waitingForItemToBePicked_) {
    return Status::kNotDone;
  }

  const auto packet = packet::building::ClientAgentActionCommandRequest::pickup(targetGlobalId_);
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  waitingForItemToBePicked_ = true;
  return Status::kNotDone;
}

} // namespace state::machine