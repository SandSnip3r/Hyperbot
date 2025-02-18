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
  if (!initialized_) {
    // Save the refId of the item we want to pick, to later verify that we got it in our inventory.
    targetRefId_ = bot_.entityTracker().getEntity(targetGlobalId_)->refObjId;
    initialized_ = true;
  }

  // At this point, we are within range of the item and can pick it up
  if (event) {
    if (sentCommand_ && waitingForItemToDespawn_) {
      if (const auto *entityDespawnedEvent = dynamic_cast<const event::EntityDespawned*>(event)) {
        if (entityDespawnedEvent->globalId == targetGlobalId_) {
          // The item we wanted to pick up despawned
          // Whether we picked it up or not doesn't matter; we're done either way
          VLOG(1) << characterNameForLog() << "The item we picked (" << bot_.gameData().getItemName(targetRefId_) << ") despawned";
          waitingForItemToDespawn_ = false;
        }
      }
    }
    if (sentCommand_ && waitingForItemToArriveInInventory_) {
      if (const auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event)) {
        if (inventoryUpdatedEvent->globalId == bot_.selfState()->globalId) {
          // Event is for us
          if (!inventoryUpdatedEvent->srcSlotNum.has_value() && inventoryUpdatedEvent->destSlotNum.has_value()) {
            // Represents a pick
            const storage::Item *item = bot_.inventory().getItem(inventoryUpdatedEvent->destSlotNum.value());
            if (item != nullptr && item->refItemId == targetRefId_) {
              // We picked up the item we wanted
              // TODO: We don't know if this is because we picked this item up, or someone else in our party picked up an item of the same type and via item distribution, we received it.
              VLOG(1) << characterNameForLog() << "The item we picked (" << bot_.gameData().getItemName(targetRefId_) << ") landed in our inventory";
              waitingForItemToArriveInInventory_ = false;
            }
          }
        }
      }
    }
    // TODO: Handle response for CommandRequest
  }

  if (sentCommand_ && !waitingForItemToDespawn_ && !waitingForItemToArriveInInventory_) {
    return Status::kDone;
  }

  if (sentCommand_ && (waitingForItemToDespawn_ || waitingForItemToArriveInInventory_)) {
    return Status::kNotDone;
  }

  if (!sentCommand_) {
    VLOG(1) << characterNameForLog() << "Sending packet to pickup " << bot_.gameData().getItemName(targetRefId_);
    const auto packet = packet::building::ClientAgentActionCommandRequest::pickup(targetGlobalId_);
    bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
    sentCommand_ = true;
    return Status::kNotDone;
  }
  throw std::runtime_error("Should be impossible to reach the end of PickItem::onUpdate()");
}

} // namespace state::machine