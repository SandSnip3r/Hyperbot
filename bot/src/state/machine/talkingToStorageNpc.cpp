#include "talkingToStorageNpc.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentActionDeselectRequest.hpp"
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentActionTalkRequest.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/clientAgentInventoryStorageOpenRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

TalkingToStorageNpc::TalkingToStorageNpc(Bot &bot) : StateMachine(bot) {
  stateMachineCreated(kName);
  // We know we are near our npc, lets find the closest npc to us.
  // TODO: This won't always work. We don't need to get very close to an npc to talk to them, we could be closer to another npc.
  npcGid_ = bot_.getClosestNpcGlobalId();
}

TalkingToStorageNpc::~TalkingToStorageNpc() {
  stateMachineDestroyed();
}

Status TalkingToStorageNpc::onUpdate(const event::Event *event) {
  if (npcInteractionState_ == NpcInteractionState::kStart) {
    // Have not yet done anything. First thing is to select the Npc
    const auto selectNpc = packet::building::ClientAgentActionSelectRequest::packet(npcGid_);
    bot_.packetBroker().injectPacket(selectNpc, PacketContainer::Direction::kBotToServer);
    // Advance state
    npcInteractionState_ = NpcInteractionState::kSelectionRequestPending;
    return Status::kNotDone;
  }

  if (npcInteractionState_ == NpcInteractionState::kSelectionRequestPending) {
    // TODO: Check if event is selection failed
    if (!bot_.selfState()->selectedEntity) {
      // Waiting for npc to be selected, nothing to do
      return Status::kNotDone;
    }

    if (*bot_.selfState()->selectedEntity != npcGid_) {
      throw std::runtime_error("We have something selected, but its not the storage npc");
    }

    // Selection succeeded
    npcInteractionState_ = NpcInteractionState::kNpcSelected;

    if (bot_.selfState()->haveOpenedStorageSinceTeleport) {
      // We have already opened our storage, we just need to talk to the npc
      const auto openStorage = packet::building::ClientAgentActionTalkRequest::packet(*bot_.selfState()->selectedEntity, packet::enums::TalkOption::kStorage);
      bot_.packetBroker().injectPacket(openStorage, PacketContainer::Direction::kBotToServer);
      npcInteractionState_ = NpcInteractionState::kShopOpenRequestPending;
    } else {
      // We have not yet opened our storage
      const auto openStorage = packet::building::ClientAgentInventoryStorageOpenRequest::packet(*bot_.selfState()->selectedEntity);
      bot_.packetBroker().injectPacket(openStorage, PacketContainer::Direction::kBotToServer);
      // TODO: Once the server responds that this is successful, the client will automatically send the packet for the talk option
      //   To be client-independent, we should block this request from the client and do this ourself
      npcInteractionState_ = NpcInteractionState::kStorageOpenRequestPending;
    }
    return Status::kNotDone;
  }

  if (bot_.selfState()->talkingGidAndOption) {
    // We are talking to an npc
    if (bot_.selfState()->talkingGidAndOption->first != npcGid_) {
      throw std::runtime_error("We're talking to some Npc, but it's not the storage Npc");
    }
    if (bot_.selfState()->talkingGidAndOption->second != packet::enums::TalkOption::kStorage) {
      throw std::runtime_error("We're talking to the storage Npc, but it's not the storage option");
    }
    npcInteractionState_ = NpcInteractionState::kShopOpened;

    // Storage is open, we're ready to store our items
    storeItems(event);

    if (npcInteractionState_ == NpcInteractionState::kShopOpened) {
      // Still storing items, dont advance
      return Status::kNotDone;
    }

    // Done storing items
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(bot_.selfState()->talkingGidAndOption->first);
    bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kBotToServer);
    return Status::kNotDone;
  } else if (npcInteractionState_ == NpcInteractionState::kShopOpenRequestPending || npcInteractionState_ == NpcInteractionState::kStorageOpenRequestPending) {
    return Status::kNotDone;
  }

  if (bot_.selfState()->selectedEntity) {
    // We have closed the shop, but still have the npc selected. Deselect them
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(*bot_.selfState()->selectedEntity);
    bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kBotToServer);
    return Status::kNotDone;
  }

  if (!bot_.selfState()->selectedEntity) {
    // Storage closed and npc deselected, completely done
    return Status::kDone;
  }
  return Status::kNotDone;
}

void TalkingToStorageNpc::storeItems(const event::Event *event) {
  // Did something just arrive in storage?
  if (event != nullptr) {
    if (dynamic_cast<const event::InventoryUpdated*>(event) != nullptr) {
      // Ignoring inventory updated events since a corresponding storage event will come with more info for us
      return;
    }
    const auto *storageUpdatedEvent = dynamic_cast<const event::StorageUpdated*>(event);
    if (storageUpdatedEvent != nullptr) {
      // At least one storage slot was updated
      pendingItemMovementRequest_ = false;
      uint8_t itemToTryStackingSlotNum;
      if (storageUpdatedEvent->destSlotNum.has_value()) {
        // Something was moved in storage. Either a deposit or an item stacked in storage
        if (storageUpdatedEvent->srcSlotNum.has_value()) {
          // This was a move within storage, a stacking in this case
          // The src slot could still contain an item, we'll try to stack that
          itemToTryStackingSlotNum = *storageUpdatedEvent->srcSlotNum;
        } else {
          // This was a deposit, try to stack the newly added item
          itemToTryStackingSlotNum = *storageUpdatedEvent->destSlotNum;
        }
      }

      if (bot_.selfState()->storage.hasItem(itemToTryStackingSlotNum)) {
        // Storage has something in this slot
        auto *item = bot_.selfState()->storage.getItem(itemToTryStackingSlotNum);
        const auto *itemExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
        if (itemExpendable != nullptr) {
          // This is a stackable item, lets see if we can stack it somewhere earlier in the storage
          const auto slotsWithThisSameItem = bot_.selfState()->storage.findItemsWithRefId(item->refItemId);
          for (auto destSlotNum : slotsWithThisSameItem) {
            if (destSlotNum >= itemToTryStackingSlotNum) {
              // Not going to try to stack this item with itself, or into a later spot
              break;
            }
            auto *destItem = bot_.selfState()->storage.getItem(destSlotNum);
            if (destItem == nullptr) {
              throw std::runtime_error("Storage said he had an item here");
            }
            const auto *destItemExpendable = dynamic_cast<const storage::ItemExpendable*>(destItem);
            if (destItemExpendable == nullptr) {
              throw std::runtime_error("Storage said this item is the same as our other expendable");
            }
            const uint16_t spaceLeftInStack = destItemExpendable->itemInfo->maxStack - destItemExpendable->quantity;
            if (spaceLeftInStack > 0) {
              // We can stack our item to this spot
              const auto moveItemInStoragePacket = packet::building::ClientAgentInventoryOperationRequest::withinStoragePacket(itemToTryStackingSlotNum, destSlotNum, std::min(itemExpendable->quantity, spaceLeftInStack), bot_.selfState()->talkingGidAndOption->first);
              bot_.packetBroker().injectPacket(moveItemInStoragePacket, PacketContainer::Direction::kBotToServer);
              // TODO: There is a potential that this function gets retriggered before the item movement completes. Maybe we ought to set an internal state to block that from messing us up
              pendingItemMovementRequest_ = true;
              return;
            }
          }
        }
      }
    }
  }

  if (pendingItemMovementRequest_) {
    // Still waiting for an item to move, nothing to do
    return;
  }

  // At this point, we werent able to stack, move the next item
  // What should we store?
  // TODO: Calculate what we can afford to store.
  std::vector<uint8_t> slotsWithItemsToStore;
  for (const auto itemTypeToStore : itemTypesToStore_) {
    const auto slotsWithThisItemType = bot_.selfState()->inventory.findItemsWithTypeId(itemTypeToStore);
    slotsWithItemsToStore.insert(slotsWithItemsToStore.end(), slotsWithThisItemType.begin(), slotsWithThisItemType.end());
  }
  // Also, try to store things which are already in storage.
  for (uint8_t slot=0; slot<bot_.selfState()->inventory.size(); ++slot) {
    if (bot_.selfState()->inventory.hasItem(slot)) {
      // Check if any slot in storage matches this item
      for (uint8_t storageSlot=0; storageSlot<bot_.selfState()->storage.size(); ++storageSlot) {
        if (bot_.selfState()->storage.hasItem(storageSlot)) {
          const auto *inventoryItem = bot_.selfState()->inventory.getItem(slot);
          const auto *storageItem = bot_.selfState()->storage.getItem(storageSlot);
          if (inventoryItem->refItemId == storageItem->refItemId) {
            // Items match. We want to store this item.
            VLOG(1) << "Want to store " << bot_.gameData().getItemName(inventoryItem->refItemId) << " because items like it are already in storage.";
            slotsWithItemsToStore.push_back(slot);
            break;
          }
        }
      }
    }
  }
  std::sort(slotsWithItemsToStore.begin(), slotsWithItemsToStore.end());
  if (!slotsWithItemsToStore.empty()) {
    // Try to store first item
    // Figure out where to store it
    const auto &slot = bot_.selfState()->storage.firstFreeSlot();
    if (slot) {
      // Have a free slot in storage
      const auto depositItemPacket = packet::building::ClientAgentInventoryOperationRequest::inventoryToStoragePacket(slotsWithItemsToStore.front(), *slot, bot_.selfState()->talkingGidAndOption->first);
      bot_.packetBroker().injectPacket(depositItemPacket, PacketContainer::Direction::kBotToServer);
      return;
    } else {
      LOG(INFO) << "Storage is full!";
      // TODO: Handle a full storage
      npcInteractionState_ = NpcInteractionState::kDoneStoring;
      return;
    }
  }

  // We didnt store anything
  npcInteractionState_ = NpcInteractionState::kDoneStoring;
}

} // namespace state::machine