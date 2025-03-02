#include "pickItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"

namespace state::machine {

PickItem::PickItem(Bot &bot, sro::scalar_types::EntityGlobalId targetGlobalId) : StateMachine(bot), targetGlobalId_(targetGlobalId) {
}

PickItem::~PickItem() {
  if (requestTimeoutEventId_) {
    bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
    requestTimeoutEventId_.reset();
  }
}

Status PickItem::onUpdate(const event::Event *event) {
  if (!initialized_) {
    // Save the refId of the item we want to pick, to later verify that we got it in our inventory.
    targetRefId_ = bot_.entityTracker().getEntity(targetGlobalId_)->refObjId;
    initialized_ = true;
  }

  // At this point, we are within range of the item and can pick it up
  if (event) {
    if (const auto *entityDespawnedEvent = dynamic_cast<const event::EntityDespawned*>(event)) {
      if (waitingForItemToDespawn_) {
        if (entityDespawnedEvent->globalId == targetGlobalId_) {
          // The item we wanted to pick up despawned
          // Whether we picked it up or not doesn't matter; we're done either way
          CHAR_VLOG(1) << "The item we picked (" << bot_.gameData().getItemName(targetRefId_) << ") despawned";
          if (requestTimeoutEventId_) {
            CHAR_VLOG(2) << "Cancelling timeout event " << *requestTimeoutEventId_;
            bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
            requestTimeoutEventId_.reset();
          }
          waitingForItemToDespawn_ = false;
          // TODO: If someone else picks it up, we'll be stuck here waiting for the update that the item has arrived in our inventory.
          //  To solve this, it would be nice to have a way to communicate back to the parent state machine that picking the item failed.
          //  Alternatively, maybe all parent state machines should be robust to pick failures.
        }
      }
    } else if (const auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event)) {
      if (waitingForItemToArriveInInventory_) {
        if (inventoryUpdatedEvent->globalId == bot_.selfState()->globalId) {
          // Event is for us
          if (!inventoryUpdatedEvent->srcSlotNum.has_value() && inventoryUpdatedEvent->destSlotNum.has_value()) {
            // Represents a pick
            const storage::Item *item = bot_.inventory().getItem(inventoryUpdatedEvent->destSlotNum.value());
            if (item != nullptr && item->refItemId == targetRefId_) {
              // We picked up the item we wanted
              // TODO: We don't know if this is because we picked this item up, or someone else in our party picked up an item of the same type and via item distribution, we received it.
              CHAR_VLOG(1) << "The item we picked (" << bot_.gameData().getItemName(targetRefId_) << ") landed in our inventory";
              if (requestTimeoutEventId_) {
                CHAR_VLOG(2) << "Cancelling timeout event " << *requestTimeoutEventId_;
                bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
                requestTimeoutEventId_.reset();
              }
              waitingForItemToArriveInInventory_ = false;
            }
          }
        }
      }
    } else if (const auto *commandErrorEvent = dynamic_cast<const event::CommandError*>(event)) {
      if (commandErrorEvent->issuingGlobalId == bot_.selfState()->globalId &&
          commandErrorEvent->command.commandType == packet::enums::CommandType::kExecute &&
          commandErrorEvent->command.actionType == packet::enums::ActionType::kPickup &&
          commandErrorEvent->command.targetGlobalId == targetGlobalId_) {
        if (requestTimeoutEventId_) {
          // We failed to pick up the item
          CHAR_VLOG(1) << "Failed to pick up " << bot_.gameData().getItemName(targetRefId_) << ". Trying again";
          bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
          requestTimeoutEventId_.reset();
        }
      }
    } else if (event->eventCode == event::EventCode::kTimeout) {
      if (requestTimeoutEventId_ && event->eventId == *requestTimeoutEventId_) {
        // Our command timed out.
        CHAR_VLOG(1) << "Command timed out";
        requestTimeoutEventId_.reset();
      }
    }
  }

  if (bot_.selfState()->bodyState() == packet::enums::BodyState::kUntouchable) {
    CHAR_VLOG(2) << "Cannot pick up item while untouchable";
    return Status::kNotDone;
  }

  if (!waitingForItemToDespawn_ && !waitingForItemToArriveInInventory_) {
    CHAR_VLOG(1) << "Item is despawned and in our inventory. Done";
    return Status::kDone;
  }

  if ((waitingForItemToDespawn_ && !waitingForItemToArriveInInventory_) ||
      (!waitingForItemToDespawn_ && waitingForItemToArriveInInventory_)) {
    CHAR_VLOG(2) << "Waiting for item to despawn and/or arrive in inventory, one of these has already happened";
    return Status::kNotDone;
  }

  // Now, it must be the case that the item is still on the ground and not in our inventory.
  if (requestTimeoutEventId_) {
    // Still waiting to see if our pick request works.
    return Status::kNotDone;
  }

  // We've either never tried to pick this item, or we tried before and nothing happened.
  tryPickItem();
  return Status::kNotDone;
}

void PickItem::tryPickItem() {
  CHAR_VLOG(1) << "Sending packet to pickup " << bot_.gameData().getItemName(targetRefId_);
  const auto packet = packet::building::ClientAgentActionCommandRequest::pickup(targetGlobalId_);
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  requestTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent(event::EventCode::kTimeout, std::chrono::milliseconds(1000));
  CHAR_VLOG(2) << "Published timeout event " << *requestTimeoutEventId_;
}

} // namespace state::machine