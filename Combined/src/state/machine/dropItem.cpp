#include "dropItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"

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
        LOG() << "Item spawned" << std::endl;
        // This is our item.
      }
    } else if (auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event); inventoryUpdatedEvent != nullptr) {
      LOG() << "inventory update" << std::endl;
      if (!inventoryUpdatedEvent->destSlotNum) {
        LOG() << "  is a drop" << std::endl;
        // Is a item drop/delete update.
        if (*inventoryUpdatedEvent->srcSlotNum == inventorySlot_) {
          LOG() << "    is our item" << std::endl;
          // Is our item.
          done_ = true;
          return;
        } else {
          LOG() << "    dropped from " << (int)*inventoryUpdatedEvent->srcSlotNum << " but we expected " << (int)inventorySlot_ << std::endl;
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
  LOG() << "Send drop packet" << std::endl;
}

bool DropItem::done() const {
  return done_;
}

} // namespace state::machine