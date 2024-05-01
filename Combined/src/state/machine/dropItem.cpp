#include "dropItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

DropItem::DropItem(Bot &bot, sro::scalar_types::StorageIndexType inventorySlot) : StateMachine(bot), inventorySlot_(inventorySlot) {
  stateMachineCreated(kName);
  const storage::Item *item = bot_.selfState().inventory.getItem(inventorySlot_);
  refId_ = item->refItemId;
}

DropItem::~DropItem() {
  stateMachineDestroyed();
}

void DropItem::onUpdate(const event::Event *event) {
  // There are two relevant events:
  //  1. Inventory event, item leaves inventory
  //  2. Entity spawned event, item hits ground.
  // Entity spawn happens before the item leaves our inventory. We will wait on the inventory update before we consider ourselves done.

  if (event) {
    if (auto *entitySpawnedEvent = dynamic_cast<const event::EntitySpawned*>(event); entitySpawnedEvent != nullptr) {
      const auto *entity = bot_.entityTracker().getEntity(entitySpawnedEvent->globalId);
      if (entity->refObjId == refId_) {
        LOG(INFO) << "Item spawned";
        // This is our item.
      }
    } else if (auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event); inventoryUpdatedEvent != nullptr) {
      LOG(INFO) << "inventory update";
      if (!inventoryUpdatedEvent->destSlotNum) {
        LOG(INFO) << "  is a drop";
        // Is a item drop/delete update.
        if (*inventoryUpdatedEvent->srcSlotNum == inventorySlot_) {
          LOG(INFO) << "    is our item";
          // Is our item.
          done_ = true;
          return;
        } else {
          LOG(INFO) << "    dropped from " << (int)*inventoryUpdatedEvent->srcSlotNum << " but we expected " << (int)inventorySlot_;
        }
      }
    }
  }

  if (waitingForItemToBeDropped_) {
    return;
  }

  const auto packet = packet::building::ClientAgentInventoryOperationRequest::dropItem(inventorySlot_);
  bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
  waitingForItemToBeDropped_ = true;
  LOG(INFO) << "Send drop packet";
}

bool DropItem::done() const {
  return done_;
}

} // namespace state::machine