#include "pickItemWithCos.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentCosCommandRequest.hpp"

namespace state::machine {

PickItemWithCos::PickItemWithCos(StateMachine *parent, sro::scalar_types::EntityGlobalId cosGlobalId, sro::scalar_types::EntityGlobalId targetGlobalId) : StateMachine(parent), cosGlobalId_(cosGlobalId), targetGlobalId_(targetGlobalId) {}

PickItemWithCos::~PickItemWithCos() {}

Status PickItemWithCos::onUpdate(const event::Event *event) {
  if (event) {
    if (const auto *entityDespawnedEvent = dynamic_cast<const event::EntityDespawned*>(event)) {
      if (entityDespawnedEvent->globalId == targetGlobalId_) {
        // The item we wanted to pick up despawned
        // Whether we picked it up or not doesn't matter; we're done either way
        return Status::kDone;
      }
    }
  }

  if (waitingForItemToBePicked_) {
    return Status::kNotDone;
  }

  const auto packet = packet::building::ClientAgentCosCommandRequest::pickup(cosGlobalId_, targetGlobalId_);
  injectPacket(packet, PacketContainer::Direction::kBotToServer);
  waitingForItemToBePicked_ = true;
  return Status::kNotDone;
}

} // namespace state::machine