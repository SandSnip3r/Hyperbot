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

  if (bot_.selfState()->bodyState() == packet::enums::BodyState::kUntouchable) {
    VLOG(1) << characterNameForLog() << "Cannot pick up item while untouchable";
    return Status::kNotDone;
  }

  // At this point, we are within range of the item and can pick it up
  if (event) {
    if (const auto *entityDespawnedEvent = dynamic_cast<const event::EntityDespawned*>(event)) {
      if (waitingForItemToDespawn_) {
        if (entityDespawnedEvent->globalId == targetGlobalId_) {
          // The item we wanted to pick up despawned
          // Whether we picked it up or not doesn't matter; we're done either way
          VLOG(1) << characterNameForLog() << "The item we picked (" << bot_.gameData().getItemName(targetRefId_) << ") despawned";
          if (requestTimeoutEventId_) {
            VLOG(2) << characterNameForLog() << "Cancelling timeout event " << *requestTimeoutEventId_;
            bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
            requestTimeoutEventId_.reset();
          }
          waitingForItemToDespawn_ = false;
          // TODO: If someone else picks it up, we'll be stuck here waiting for the update that the item has arrived in our inventory.
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
              VLOG(1) << characterNameForLog() << "The item we picked (" << bot_.gameData().getItemName(targetRefId_) << ") landed in our inventory";
              if (requestTimeoutEventId_) {
                VLOG(2) << characterNameForLog() << "Cancelling timeout event " << *requestTimeoutEventId_;
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
          commandErrorEvent->command.actionType == packet::enums::ActionType::kPickup) {
        if (requestTimeoutEventId_) {
          // We failed to pick up the item
          VLOG(1) << characterNameForLog() << "Failed to pick up " << bot_.gameData().getItemName(targetRefId_) << ". Trying again";
          bot_.eventBroker().cancelDelayedEvent(*requestTimeoutEventId_);
          requestTimeoutEventId_.reset();
          tryPickItem();
          return Status::kNotDone;
        }
      }
    } else if (event->eventCode == event::EventCode::kTimeout) {
      if (requestTimeoutEventId_ && event->eventId == *requestTimeoutEventId_) {
        // Our command timed out.
        VLOG(1) << characterNameForLog() << "Command timed out";
        requestTimeoutEventId_.reset();
        tryPickItem();
        return Status::kNotDone;
      }
    }
    // TODO: Handle response for CommandRequest
  }

  if (!waitingForItemToDespawn_ && !waitingForItemToArriveInInventory_) {
    VLOG(1) << characterNameForLog() << "Item is despawned and in our inventory. Done";
    return Status::kDone;
  }

  if (requestTimeoutEventId_ && (waitingForItemToDespawn_ || waitingForItemToArriveInInventory_)) {
    return Status::kNotDone;
  }

  if (!requestTimeoutEventId_) {
    tryPickItem();
    return Status::kNotDone;
  }
  throw std::runtime_error("Should be impossible to reach the end of PickItem::onUpdate()");
}

void PickItem::tryPickItem() {
  VLOG(1) << characterNameForLog() << "Sending packet to pickup " << bot_.gameData().getItemName(targetRefId_);
  const auto packet = packet::building::ClientAgentActionCommandRequest::pickup(targetGlobalId_);
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  requestTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(888), event::EventCode::kTimeout);
  VLOG(2) << characterNameForLog() << "Published timeout event " << *requestTimeoutEventId_;
}

} // namespace state::machine