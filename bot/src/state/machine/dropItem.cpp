#include "dropItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

DropItem::DropItem(StateMachine *parent, sro::scalar_types::StorageIndexType inventorySlot) : StateMachine(parent), inventorySlot_(inventorySlot) {
  std::shared_ptr<entity::Self> selfEntity = bot_.selfState();
  const storage::Item *item = selfEntity->inventory.getItem(inventorySlot_);
  refId_ = item->refItemId;
}

DropItem::~DropItem() {}

Status DropItem::onUpdate(const event::Event *event) {
  // There are two relevant events:
  //  1. Inventory event, item leaves inventory
  //  2. Entity spawned event, item hits ground.
  // Entity spawn happens before the item leaves our inventory. We will wait on the inventory update before we consider ourselves done.

  if (event) {
    if (auto *entitySpawnedEvent = dynamic_cast<const event::EntitySpawned*>(event); entitySpawnedEvent != nullptr) {
      std::shared_ptr<const entity::Entity> entity = bot_.worldState().getEntity(entitySpawnedEvent->globalId);
      if (entity->refObjId == refId_) {
        LOG(INFO) << "Item spawned";
        // This is our item.
      }
    } else if (auto *itemMovedEvent = dynamic_cast<const event::ItemMoved*>(event); itemMovedEvent != nullptr) {
      if (itemMovedEvent->globalId == bot_.selfState()->globalId) {
        LOG(INFO) << "inventory update";
        if (!itemMovedEvent->destination && itemMovedEvent->source && itemMovedEvent->source->storage == sro::storage::Storage::kInventory) {
          const sro::scalar_types::StorageIndexType srcSlotNum = itemMovedEvent->source->slotNum;
          LOG(INFO) << "  is a drop";
          // Is a item drop/delete update.
          if (srcSlotNum == inventorySlot_) {
            LOG(INFO) << "    is our item";
            // Is our item.
            return Status::kDone;
          } else {
            LOG(INFO) << "    dropped from " << (int)srcSlotNum << " but we expected " << (int)inventorySlot_;
          }
        }
      }
    }
  }

  if (waitingForItemToBeDropped_) {
    return Status::kNotDone;
  }

  const auto packet = packet::building::ClientAgentInventoryOperationRequest::dropItem(inventorySlot_);
  injectPacket(packet, PacketContainer::Direction::kBotToServer);
  waitingForItemToBeDropped_ = true;
  LOG(INFO) << "Send drop packet";
  return Status::kNotDone;
}

} // namespace state::machine