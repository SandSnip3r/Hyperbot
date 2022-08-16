#include "stateMachine.hpp"

#include "bot.hpp"
#include "logging.hpp"
#include "math/position.hpp"
#include "packet/building/clientAgentActionDeselectRequest.hpp"
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentActionTalkRequest.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/clientAgentInventoryRepairRequest.hpp"
#include "packet/building/clientAgentInventoryStorageOpenRequest.hpp"
#include "packet/building/serverAgentInventoryOperationResponse.hpp"

namespace state::machine {

CommonStateMachine::CommonStateMachine(Bot &bot) : bot_(bot) {
}

void CommonStateMachine::pushBlockedOpcode(packet::Opcode opcode) {
  if (!bot_.proxy_.blockingOpcode(opcode)) {
    bot_.proxy_.blockOpcode(opcode);
    blockedOpcodes_.push_back(opcode);
  }
}

CommonStateMachine::~CommonStateMachine() {
  // Undo all blocked opcodes
  for (const auto opcode : blockedOpcodes_) {
    bot_.proxy_.unblockOpcode(opcode);
  }
}

// =====================================================================================================================================
// ===============================================================Walking===============================================================
// =====================================================================================================================================

Walking::Walking(Bot &bot, const std::vector<packet::structures::Position> &waypoints) : bot_(bot), waypoints_(waypoints) {
  if (waypoints_.empty()) {
    throw std::runtime_error("Given empty list of waypoints");
  }
}

void Walking::onUpdate(const event::Event *event) {
  if (bot_.selfState_.moving()) {
    // Still moving, nothing to do
    return;
  }

  // We're not moving
  // Did we just arrive at this waypoint?
  if (math::position::calculateDistance(bot_.selfState_.position(), waypoints_[currentWaypointIndex_]) < 5) { // TODO: Choose better measure of "close enough"
    // Just arrived at the waypoint, increment index
    ++currentWaypointIndex_;
    requestedMovement_ = false;
  } else if (requestedMovement_) {
    // Already asked to move, nothing to do
    return;
  }

  // We are not moving, and we do not have a pending movement request
  if (done()) {
    // Finished walking
    return;
  }

  // We are not moving, we're not at the current waypoint, and there's not a pending movement request
  // Send a request to move to the current waypoint
  const auto &currentWaypoint = waypoints_[currentWaypointIndex_];
  const auto movementPacket = packet::building::ClientAgentCharacterMoveRequest::packet(currentWaypoint.regionId, currentWaypoint.xOffset, currentWaypoint.yOffset, currentWaypoint.zOffset);
  bot_.broker_.injectPacket(movementPacket, PacketContainer::Direction::kClientToServer);
  requestedMovement_ = true;
}

bool Walking::done() const {
  return (currentWaypointIndex_ == waypoints_.size());
}

// =====================================================================================================================================
// =============================================================BuyingItems=============================================================
// =====================================================================================================================================

BuyingItems::BuyingItems(Bot &bot, const std::map<uint32_t, PurchaseRequest> &itemsToBuy) : CommonStateMachine(bot), itemsToBuy_(itemsToBuy) {
  // We must be talking to an NPC at this point
  // Prevent the client from closing the talk dialog
  pushBlockedOpcode(packet::Opcode::kClientAgentActionDeselectRequest);
  // Prevent the client from moving items in inventory
  pushBlockedOpcode(packet::Opcode::kClientAgentInventoryOperationRequest);
}

void BuyingItems::onUpdate(const event::Event *event) {
  if (event) {
    if (auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event)) {
      if (inventoryUpdatedEvent->destSlotNum) {
        // TODO: We dont actually know if this was our purchase, for now, we assume it was
        if (inventoryUpdatedEvent->srcSlotNum) {
          // This was a stacking
          waitingOnItemMovementResponse_ = false;
        } else {
          // TODO: Make sure this was our purchase
          //  This could have been a result from another method of aquiring an item.
          //  ex. A pickup by a party member

          // Purchase was successful. Adjust shopping list to reflect the newly desired quantity
          const auto *itemAtInventorySlot = bot_.selfState().inventory.getItem(*inventoryUpdatedEvent->destSlotNum);
          if (itemAtInventorySlot == nullptr) {
            throw std::runtime_error("Got an item from our inventory, but there's nothing here");
          }
          auto it = itemsToBuy_.find(itemAtInventorySlot->refItemId);
          if (it == itemsToBuy_.end()) {
            throw std::runtime_error("Thought we bought an item, but its not in our to-buy list");
          }
          const auto beforeCount = it->second.quantity;
          uint16_t countBought;
          if (const auto *itemExp = dynamic_cast<const storage::ItemExpendable*>(itemAtInventorySlot)) {
            countBought = itemExp->quantity;
          } else {
            countBought = 1;
          }
          if (countBought > it->second.quantity) {
            throw std::runtime_error("Somehow bought more than we wanted to");
          }
          it->second.quantity -= std::min(countBought, it->second.quantity);

          if (it->second.quantity == 0) {
            // No more of these to buy, delete from shopping list
            itemsToBuy_.erase(it);
          }
          waitingOnBuyResponse_ = false;

          // We successfully blocked the server's purchase response from reaching the client, unblock that packet type
          bot_.proxy().unblockOpcode(packet::Opcode::kServerAgentInventoryOperationResponse);
          // Since we blocked the packet which tells the client about this purchase, we need to spoof an item spawning in the character's inventory
          const auto itemBuySpoofPacket = packet::building::ServerAgentInventoryOperationResponse::addItemByServerPacket(*inventoryUpdatedEvent->destSlotNum, *itemAtInventorySlot);
          bot_.packetBroker().injectPacket(itemBuySpoofPacket, PacketContainer::Direction::kServerToClient);
        }

        if (bot_.selfState().inventory.hasItem(*inventoryUpdatedEvent->destSlotNum)) {
          // Now, lets see if we want to stack this item. It could have been just bought, or we just stacked some of it into another slot
          const auto *itemAtInventorySlot = bot_.selfState().inventory.getItem(*inventoryUpdatedEvent->destSlotNum);
          if (itemAtInventorySlot == nullptr) {
            throw std::runtime_error("Got an item from our inventory, but there's nothing here");
          }
          if (const auto *destItemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(itemAtInventorySlot)) {
            const auto refIdToStack = itemAtInventorySlot->refItemId;
            auto inventorySlotsWithThisItem = bot_.selfState().inventory.findItemsWithRefId(refIdToStack);
            if (inventorySlotsWithThisItem.size() > 1) {
              // Try to stack backwards
              std::reverse(inventorySlotsWithThisItem.begin(), inventorySlotsWithThisItem.end());
              bool stackedAnItem{false};
              for (int i=0; i<inventorySlotsWithThisItem.size(); ++i) {
                const auto laterItemIndex = inventorySlotsWithThisItem[i];
                for (int j=i+1; j<inventorySlotsWithThisItem.size(); ++j) {
                  const auto earlierItemIndex = inventorySlotsWithThisItem[j];
                  if (!bot_.selfState().inventory.hasItem(earlierItemIndex)) {
                    throw std::runtime_error("We were told there was an item here");
                  }
                  const auto *earlierItem = bot_.selfState().inventory.getItem(earlierItemIndex);
                  const auto *earlierItemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(earlierItem);
                  if (earlierItemAsExpendable == nullptr) {
                    throw std::runtime_error("This item must be an expendable");
                  }
                  if (earlierItemAsExpendable->quantity == earlierItemAsExpendable->itemInfo->maxStack) {
                    // Stack is already full, cannot stack to here
                    continue;
                  }
                  // We can stack the item in slot laterItemIndex to slot earlierItemIndex
                  const auto spaceLeftInStack = earlierItemAsExpendable->itemInfo->maxStack - earlierItemAsExpendable->quantity;
                  if (!bot_.selfState().inventory.hasItem(laterItemIndex)) {
                    throw std::runtime_error("We were told there was an item here");
                  }
                  const auto *laterItem = bot_.selfState().inventory.getItem(laterItemIndex);
                  const auto *laterItemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(laterItem);
                  if (laterItemAsExpendable == nullptr) {
                    throw std::runtime_error("This item must be an expendable");
                  }
                  // Stack item
                  const auto moveItemInInventoryPacket = packet::building::ClientAgentInventoryOperationRequest::withinInventoryPacket(laterItemIndex, earlierItemIndex, std::min(laterItemAsExpendable->quantity, static_cast<uint16_t>(spaceLeftInStack)));
                  bot_.packetBroker().injectPacket(moveItemInInventoryPacket, PacketContainer::Direction::kClientToServer);
                  waitingOnItemMovementResponse_ = true;
                  stackedAnItem = true;
                  break;
                }
                if (stackedAnItem) {
                  // Dont try stacking multiple in one go
                  break;
                }
              }
            }
          }
        }
      }
    }
  }

  if (waitingOnItemMovementResponse_) {
    // Waiting on a stacking, nothing to do
    return;
  }

  if (waitingOnBuyResponse_) {
    // Waiting on an item we bought, nothing to do
    return;
  }

  if (itemsToBuy_.empty()) {
    // Nothing else to buy
    done_ = true;
    return;
  }

  const auto &nextPurchaseRequest = itemsToBuy_.begin()->second;
  const auto countToBuy = std::min(nextPurchaseRequest.quantity, static_cast<uint16_t>(nextPurchaseRequest.maxStackSize));
  // Block the server's response from reaching the client
  bot_.proxy().blockOpcode(packet::Opcode::kServerAgentInventoryOperationResponse);
  const auto buyItemPacket = packet::building::ClientAgentInventoryOperationRequest::buyPacket(nextPurchaseRequest.tabIndex, nextPurchaseRequest.itemIndex, countToBuy, bot_.selfState().talkingGidAndOption->first);
  bot_.packetBroker().injectPacket(buyItemPacket, PacketContainer::Direction::kClientToServer);
  {
    // TODO: We wouldn't have to do this if we could subscribe to our injected packets (via the PacketBroker)
    packet::structures::ItemMovement itemMovement;
    itemMovement.type = packet::enums::ItemMovementType::kBuyFromNPC;
    itemMovement.globalId = bot_.selfState().talkingGidAndOption->first;
    itemMovement.storeTabNumber = nextPurchaseRequest.tabIndex;
    itemMovement.storeSlotNumber = nextPurchaseRequest.itemIndex;
    bot_.selfState().setUserPurchaseRequest(itemMovement);
  }
  waitingOnBuyResponse_ = true;
}

bool BuyingItems::done() const {
  return done_;
}

// =====================================================================================================================================
// =========================================================TalkingToStorageNpc=========================================================
// =====================================================================================================================================


TalkingToStorageNpc::TalkingToStorageNpc(Bot &bot) : bot_(bot) {
  // Figure out what we want to deposit into storage
  const uint16_t kArrowTypeId{helpers::type_id::makeTypeId(3,3,4,1)};
  const uint16_t kHpPotionTypeId{helpers::type_id::makeTypeId(3,3,1,1)};
  itemTypesToStore_.insert(kArrowTypeId);
  itemTypesToStore_.insert(kHpPotionTypeId);
}

void TalkingToStorageNpc::onUpdate(const event::Event *event) {
  if (npcInteractionState_ == NpcInteractionState::kStart) {
    // Have not yet done anything. First thing is to select the Npc
    const auto selectNpc = packet::building::ClientAgentActionSelectRequest::packet(kStorageNpcGId);
    bot_.broker_.injectPacket(selectNpc, PacketContainer::Direction::kClientToServer);
    // Advance state
    npcInteractionState_ = NpcInteractionState::kSelectionRequestPending;
    return;
  }

  if (npcInteractionState_ == NpcInteractionState::kSelectionRequestPending) {
    // TODO: Check if event is selection failed
    if (!bot_.selfState_.selectedEntity) {
      // Waiting for npc to be selected, nothing to do
      return;
    }

    if (*bot_.selfState_.selectedEntity != kStorageNpcGId) {
      throw std::runtime_error("We have something selected, but its not the storage npc");
    }

    // Selection succceeded
    npcInteractionState_ = NpcInteractionState::kNpcSelected;

    if (bot_.selfState_.haveOpenedStorageSinceTeleport) {
      // We have already opened our storage, we just need to talk to the npc
      const auto openStorage = packet::building::ClientAgentActionTalkRequest::packet(*bot_.selfState_.selectedEntity, packet::enums::TalkOption::kStorage);
      bot_.broker_.injectPacket(openStorage, PacketContainer::Direction::kClientToServer);
      {
        // TODO: We wouldn't have to do this if we could subscribe to our injected packets (via the PacketBroker)
        bot_.selfState_.pendingTalkGid = *bot_.selfState_.selectedEntity;
      }
      npcInteractionState_ = NpcInteractionState::kShopOpenRequestPending;
    } else {
      // We have not yet opened our storage
      const auto openStorage = packet::building::ClientAgentInventoryStorageOpenRequest::packet(*bot_.selfState_.selectedEntity);
      bot_.broker_.injectPacket(openStorage, PacketContainer::Direction::kClientToServer);
      // TODO: Once the server responds that this is successful, the client will automatically send the packet for the talk option
      //   To be client-independent, we should block this request from the client and do this ourself
      npcInteractionState_ = NpcInteractionState::kStorageOpenRequestPending;
    }
    return;
  }
  
  if (bot_.selfState_.talkingGidAndOption) {
    // We are talking to an npc
    if (bot_.selfState_.talkingGidAndOption->first != kStorageNpcGId) {
      throw std::runtime_error("We're talking to some Npc, but it's not the storage Npc");
    }
    if (bot_.selfState_.talkingGidAndOption->second != packet::enums::TalkOption::kStorage) {
      throw std::runtime_error("We're talking to the storage Npc, but it's not the storage option");
    }
    npcInteractionState_ = NpcInteractionState::kShopOpened;

    // Storage is open, we're ready to store our items
    storeItems(event);

    if (npcInteractionState_ == NpcInteractionState::kShopOpened) {
      // Still storing items, dont advance
      return;
    }

    // Done storing items
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(bot_.selfState_.talkingGidAndOption->first);
    bot_.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
    return;
  } else if (npcInteractionState_ == NpcInteractionState::kShopOpenRequestPending || npcInteractionState_ == NpcInteractionState::kStorageOpenRequestPending) {
    return;
  }
  
  if (bot_.selfState_.selectedEntity) {
    // We have closed the shop, but still have the npc selected. Deselect them
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(*bot_.selfState_.selectedEntity);
    bot_.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
    return;
  }

  if (!bot_.selfState_.selectedEntity) {
    // Storage closed and npc deselected, completely done
    done_ = true;
    return;
  }
}

bool TalkingToStorageNpc::done() const {
  return done_;
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

      if (bot_.selfState_.storage.hasItem(itemToTryStackingSlotNum)) {
        // Storage has something in this slot
        auto *item = bot_.selfState_.storage.getItem(itemToTryStackingSlotNum);
        const auto *itemExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
        if (itemExpendable != nullptr) {
          // This is a stackable item, lets see if we can stack it somewhere earlier in the storage
          const auto slotsWithThisSameItem = bot_.selfState_.storage.findItemsWithRefId(item->refItemId);
          for (auto destSlotNum : slotsWithThisSameItem) {
            if (destSlotNum >= itemToTryStackingSlotNum) {
              // Not going to try to stack this item with itself, or into a later spot
              break;
            }
            auto *destItem = bot_.selfState_.storage.getItem(destSlotNum);
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
              const auto moveItemInStoragePacket = packet::building::ClientAgentInventoryOperationRequest::withinStoragePacket(itemToTryStackingSlotNum, destSlotNum, std::min(itemExpendable->quantity, spaceLeftInStack), bot_.selfState_.talkingGidAndOption->first);
              bot_.broker_.injectPacket(moveItemInStoragePacket, PacketContainer::Direction::kClientToServer);
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
  std::vector<uint8_t> slotsWithItemsToStore;
  for (const auto itemTypeToStore : itemTypesToStore_) {
    const auto slotsWithThisItemType = bot_.selfState_.inventory.findItemsWithTypeId(itemTypeToStore);
    slotsWithItemsToStore.insert(slotsWithItemsToStore.end(), slotsWithThisItemType.begin(), slotsWithThisItemType.end());
  }
  std::sort(slotsWithItemsToStore.begin(), slotsWithItemsToStore.end());
  if (!slotsWithItemsToStore.empty()) {
    // Try to store first item
    // Figure out where to store it
    const auto &slot = bot_.selfState_.storage.firstFreeSlot();
    if (slot) {
      // Have a free slot in storage
      const auto depositItemPacket = packet::building::ClientAgentInventoryOperationRequest::inventoryToStoragePacket(slotsWithItemsToStore.front(), *slot, bot_.selfState_.talkingGidAndOption->first);
      bot_.broker_.injectPacket(depositItemPacket, PacketContainer::Direction::kClientToServer);
      return;
    } else {
      LOG() << "Storage is full!" << std::endl;
      // TODO: Handle a full storage
      npcInteractionState_ = NpcInteractionState::kDoneStoring;
      return;
    }
  }

  // We didnt store anything
  npcInteractionState_ = NpcInteractionState::kDoneStoring;
}

// =====================================================================================================================================
// ==========================================================TalkingToShopNpc===========================================================
// =====================================================================================================================================

TalkingToShopNpc::TalkingToShopNpc(Bot &bot, Npc npc, const std::map<uint32_t, int> &shoppingList) : CommonStateMachine(bot), npc_(npc), shoppingList_(shoppingList) {
  // We know we are near our npc, lets find the closest npc to us
  npcGid_ = [&]{
    std::optional<uint32_t> closestNpcGId;
    float closestNpcDistance = std::numeric_limits<float>::max();
    const auto &entityMap = bot_.entityState_.getEntityMap();
    for (const auto &entityIdObjectPair : entityMap) {
      const auto &objectPtr = entityIdObjectPair.second;
      if (!objectPtr) {
        throw std::runtime_error("Entity map contains a null item");
      }
      if (objectPtr->type != packet::parsing::ObjectType::kNonplayerCharacter) {
        // Not an npc, skip
        continue;
      }

      const packet::structures::Position npcPosition{ objectPtr->regionId, objectPtr->x, objectPtr->y, objectPtr->z };
      const auto distanceToNpc = math::position::calculateDistance(bot_.selfState_.position(), npcPosition);
      if (distanceToNpc < closestNpcDistance) {
        closestNpcGId = entityIdObjectPair.first;
        closestNpcDistance = distanceToNpc;
      }
    }
    if (!closestNpcGId) {
      throw std::runtime_error("There is no NPC within range, weird");
    }
    return *closestNpcGId;
  }();

  // Figure out what items to items to buy
  figureOutWhatToBuy();

  pushBlockedOpcode(packet::Opcode::kServerAgentActionSelectResponse);
  pushBlockedOpcode(packet::Opcode::kServerAgentActionTalkResponse);
  pushBlockedOpcode(packet::Opcode::kClientAgentInventoryOperationRequest);
}

void TalkingToShopNpc::figureOutWhatToBuy() {
  for (const auto &shoppingItemIdCountPair : shoppingList_) {
    const auto itemRefId = shoppingItemIdCountPair.first;
    // Do we have enough of these in our inventory?
    const auto slotsWithItem = bot_.selfState_.inventory.findItemsWithRefId(itemRefId);
    int ownedCountOfItem{0};
    bool haveEnoughOfThisItem{false};
    for (const auto slotWithItem : slotsWithItem) {
      const auto *itemPtr = bot_.selfState_.inventory.getItem(slotWithItem);
      if (itemPtr == nullptr) {
        throw std::runtime_error("Inventory said we had an item, but it is null");
      }
      if (const auto *expendableItemPtr = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) {
        ownedCountOfItem += expendableItemPtr->quantity;
      } else {
        ++ownedCountOfItem;
      }
      if (ownedCountOfItem >= shoppingItemIdCountPair.second) {
        haveEnoughOfThisItem = true;
        break;
      }
    }
    if (haveEnoughOfThisItem) {
      continue;
    }

    if (!bot_.gameData_.itemData().haveItemWithId(itemRefId)) {
      throw std::runtime_error("Want to buy an item for which we have no data");
    }
    const auto &item = bot_.gameData_.itemData().getItemById(itemRefId);
    const auto &nameOfItemToBuy = item.codeName128;
    const auto *npc = bot_.entityState_.getEntity(npcGid_);
    if (npc == nullptr) {
      throw std::runtime_error("Got entity, but it's null");
    }
    if (npc->type != packet::parsing::ObjectType::kNonplayerCharacter) {
      throw std::runtime_error("Entity is not a NonplayerCharacter");
    }

    if (!bot_.gameData_.characterData().haveCharacterWithId(npc->refObjId)) {
      throw std::runtime_error("Don't have character data for this Npc");
    }
    const auto &character = bot_.gameData_.characterData().getCharacterById(npc->refObjId);
    auto &shopTabs = bot_.gameData_.shopData().getNpcTabs(character.codeName128);
    bool foundItemInShop{false};
    for (int tabIndex=0; tabIndex<shopTabs.size(); ++tabIndex) {
      const auto &tab = shopTabs[tabIndex];
      const auto &packageMap = tab.getPackageMap();
      for (const auto &itemIndexAndScrapPair : packageMap) {
        if (itemIndexAndScrapPair.second.refItemCodeName == nameOfItemToBuy) {
          itemsToBuy_[itemRefId] = BuyingItems::PurchaseRequest{ static_cast<uint8_t>(tabIndex), itemIndexAndScrapPair.first, static_cast<uint16_t>(shoppingItemIdCountPair.second - ownedCountOfItem), item.maxStack };
          foundItemInShop = true;
          break;
        }
      }
      if (foundItemInShop) {
        break;
      }
    }
  }
}

bool TalkingToShopNpc::doneBuyingItems() const {
  if (itemsToBuy_.empty()) {
    return true;
  }
  auto *buyingItemsState = std::get_if<BuyingItems>(&childState_);
  return (buyingItemsState != nullptr) && buyingItemsState->done();
}

bool TalkingToShopNpc::needToRepair() const {
  if (npc_ != Npc::kBlacksmith && npc_ != Npc::kProtector) {
    return false;
  }
  for (int i=0; i<bot_.selfState().inventory.size(); ++i) {
    if (!bot_.selfState().inventory.hasItem(i)) {
      // Nothing in this slot
      continue;
    }
    const auto *itemPtr = bot_.selfState().inventory.getItem(i);
    if (!itemPtr->itemInfo->canRepair) {
      // Not a repairable item
      continue;
    }
    const auto *itemAsEquip = dynamic_cast<const storage::ItemEquipment*>(itemPtr);
    if (itemAsEquip == nullptr) {
      LOG() << "Item can be repaired, but it's not an equipment, weird" << std::endl;
      continue;
    }
    if (itemAsEquip->repairInvalid(bot_.gameData_)) {
      continue;
    }
    if (itemAsEquip->durability < itemAsEquip->maxDurability(bot_.gameData_)) {
      // This item can be repaired
      return true;
    }
  }

  // Nothing to repair
  return false;
}

bool TalkingToShopNpc::doneWithNpc() const {
  return doneBuyingItems() && !needToRepair();
}

void TalkingToShopNpc::onUpdate(const event::Event *event) {
  if (done_) {
    LOG() << "TalkingToShopNpc on update called, but we're done. This is a smell of imperfect logic" << std::endl;
    return;
  }

  if (bot_.selfState_.talkingGidAndOption) {
    // We are talking to an Npc
    if (bot_.selfState_.talkingGidAndOption->first != npcGid_) {
      throw std::runtime_error("We're not talking to the potion Npc that we thought we were");
    }
    if (bot_.selfState_.talkingGidAndOption->second != packet::enums::TalkOption::kStore) {
      throw std::runtime_error("We're not in the talk option that we thought we were");
    }
    if (waitingForTalkResponse_) {
      // Successfully began talking to Npc
      waitingForTalkResponse_ = false;
      childState_.emplace<BuyingItems>(bot_, itemsToBuy_);
    }

    // Now that we are talking to the npc, start buying items
    auto *buyingItemsState = std::get_if<BuyingItems>(&childState_);
    if (buyingItemsState == nullptr) {
      throw std::runtime_error("If we reach this point, the state must be BuyingItems");
    }
    buyingItemsState->onUpdate(event);
    if (!buyingItemsState->done()) {
      // Still buying items, do not continue
      return;
    }

    // Done buying items at this point
    if (waitingOnStopTalkResponse_) {
      // Already deselected to close the shop, nothing else to do
      return;
    }

    // Close the shop
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(bot_.selfState_.talkingGidAndOption->first);
    bot_.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
    waitingOnStopTalkResponse_ = true;
    return;
  } else {
    // We are not talking to an npc
    if (bot_.selfState_.selectedEntity) {
      // We have something selected
      if (*bot_.selfState_.selectedEntity != npcGid_) {
        throw std::runtime_error("We have something selected, but it's not our expected Npc");
      }
      if (waitingForSelectionResponse_) {
        waitingForSelectionResponse_ = false;
      }

      // We're either about to buy items or done buying items
      if (doneBuyingItems()) {
        // We must deselect the npc
        if (waitingOnDeselectionResponse_) {
          // Already deselected, nothing to do
          return;
        }

        // Delect the npc
        const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(*bot_.selfState_.selectedEntity);
        bot_.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
        waitingOnDeselectionResponse_ = true;
        return;
      } else {
        // We must talk to the npc
        if (waitingForTalkResponse_) {
          // Already requested talk, nothing to do
          return;
        }

        // Talk to npc
        const auto openStorage = packet::building::ClientAgentActionTalkRequest::packet(*bot_.selfState_.selectedEntity, packet::enums::TalkOption::kStore);
        bot_.broker_.injectPacket(openStorage, PacketContainer::Direction::kClientToServer);
        waitingForTalkResponse_ = true;

        {
          // TODO: We wouldn't have to do this if we could subscribe to our injected packets (via the PacketBroker)
          bot_.selfState_.pendingTalkGid = *bot_.selfState_.selectedEntity;
        }
      }
    } else {
      // No Npc is selected
      if (doneWithNpc()) {
        done_ = true;
        return;
      }

      // Not done with npc, repair before considering opening npc
      if (needToRepair()) {
        if (waitingForRepairResponse_) {
          // Still waiting on repair response, dont send another
          return;
        }

        // Repair
        const auto repairAllPacket = packet::building::ClientAgentInventoryRepairRequest::repairAllPacket(npcGid_);
        bot_.broker_.injectPacket(repairAllPacket, PacketContainer::Direction::kClientToServer);
        waitingForRepairResponse_ = true;
        return;
      } else if (waitingForRepairResponse_) {
        // Dont need to repair anymore, request was successful
        waitingForRepairResponse_ = false;
      }

      // Not done with npc and dont need to repair, we must need to buy items
      if (waitingForSelectionResponse_) {
        // Already requested selection, nothing to do
        return;
      }

      // Select Npc
      const auto selectNpc = packet::building::ClientAgentActionSelectRequest::packet(npcGid_);
      bot_.broker_.injectPacket(selectNpc, PacketContainer::Direction::kClientToServer);
      waitingForSelectionResponse_ = true;
    }
  }
}

bool TalkingToShopNpc::done() const {
  return done_;
}

// =====================================================================================================================================
// =============================================================Townlooping=============================================================
// =====================================================================================================================================

Townlooping::Townlooping(Bot &bot) : CommonStateMachine(bot) {
  // Build a shopping list
  shoppingList_ = {
    { 8, 200 }, //ITEM_ETC_HP_POTION_05 (XL hp potion)
    { 15, 200 }, //ITEM_ETC_MP_POTION_05 (XL mp potion)
    { 59, 100 }, //ITEM_ETC_CURE_ALL_05 (M special universal pill)
    { 10377, 50 }, //ITEM_ETC_CURE_RANDOM_04 (XL purification pill)
    { 2198, 50 }, //ITEM_ETC_SCROLL_RETURN_02 (Special Return Scroll)
    { 62, 1000 }, //ITEM_ETC_AMMO_ARROW_01 (Arrow)
    { 3909, 1 }, //ITEM_COS_C_DHORSE1 (Ironclad Horse)
  };
  // Figure out which npcs we want to visit and in what order
  npcsToVisit_ = { Npc::kStorage, Npc::kPotion , Npc::kGrocery, Npc::kBlacksmith, Npc::kProtector, Npc::kStable };
  // Calculate the path to the first Npc
  std::vector<packet::structures::Position> pathToFirstNpc = {
    {25000, 981.0f, -32.0f, 1032.0f}
  };
  // Initialize state as walking
  childState_.emplace<Walking>(bot_, pathToFirstNpc);
}

void Townlooping::onUpdate(const event::Event *event) {
TODO_REMOVE_THIS_LABEL:
  if (done()) {
    return;
  }
  if (auto *walkingState = std::get_if<Walking>(&childState_)) {
    walkingState->onUpdate(event);

    if (walkingState->done()) {
      // Done walking, advance state
      childState_.emplace<TalkingToNpc>();
      auto &talkingToNpcState = std::get<TalkingToNpc>(childState_);
      if (npcsToVisit_[currentNpcIndex_] == Npc::kStorage) {
        talkingToNpcState.emplace<TalkingToStorageNpc>(bot_);
      } else {
        talkingToNpcState.emplace<TalkingToShopNpc>(bot_, npcsToVisit_[currentNpcIndex_], shoppingList_);
      }
      // TODO: Go back to the top of this function
      goto TODO_REMOVE_THIS_LABEL;
    }
  } else if (auto *talkingToNpcState = std::get_if<TalkingToNpc>(&childState_)) {
    bool doneTalkingToNpc{false};
    if (auto *talkingToStorageNpcState = std::get_if<TalkingToStorageNpc>(talkingToNpcState)) {
      talkingToStorageNpcState->onUpdate(event);
      doneTalkingToNpc = talkingToStorageNpcState->done();
    } else if (auto *talkingToShopNpcState = std::get_if<TalkingToShopNpc>(talkingToNpcState)) {
      talkingToShopNpcState->onUpdate(event);
      doneTalkingToNpc = talkingToShopNpcState->done();
    }

    if (doneTalkingToNpc) {
      // Moving on to next npc
      ++currentNpcIndex_;
      if (done()) {
        // No more Npcs, done with townloop
        LOG() << "No more npcs to visit, done with townloop" << std::endl;
        return;
      }

      // Calculate the path from the just-finished npc to the next npc
      auto path = pathBetweenNpcs(npcsToVisit_[currentNpcIndex_-1], npcsToVisit_[currentNpcIndex_]);
      // Update our state to walk to the next npc
      childState_.emplace<Walking>(bot_, path);
      // TODO: Go back to the top of this function
      goto TODO_REMOVE_THIS_LABEL;
    }
  }
}

bool Townlooping::done() const {
  return (currentNpcIndex_ == npcsToVisit_.size());
}

std::vector<packet::structures::Position> Townlooping::pathBetweenNpcs(Npc npcSrc, Npc npcDest) const {
  using PathType = std::vector<packet::structures::Position>;
  static const std::map<Npc, std::map<Npc, PathType>> pathBetween = []{
    std::map<Npc, std::map<Npc, PathType>> result;

    result[Npc::kStorage][Npc::kPotion] = {{ 25000, 1525.0f, 0.0f, 1385.0f }};
    result[Npc::kStorage][Npc::kBlacksmith] = {{ 25000, 397.0f, 0.0f, 1358.0f }};
    result[Npc::kStorage][Npc::kProtector] = {{ 25000, 363.0f, 0.0f, 1083.0f }};
    result[Npc::kStorage][Npc::kStable] = {{ 25000, 390.0f, 0.0f, 493.0f }};
    result[Npc::kStorage][Npc::kGrocery] = {{ 25000, 1618.0f, 0.0f, 1078.0f }};

    result[Npc::kGrocery][Npc::kPotion] = {{ 25000, 1525.0f, 0.0f, 1385.0f }};
    result[Npc::kGrocery][Npc::kProtector] = {{ 25000, 363.0f, 0.0f, 1083.0f }};
    result[Npc::kGrocery][Npc::kBlacksmith] = {{ 25000, 910.0f, -32.0f, 1148.0f }, { 25000, 397.0f, 0.0f, 1358.0f }};
    result[Npc::kGrocery][Npc::kStable] = {{ 25000, 1456.0f, 0.0f, 1025.0f }, { 25000, 1020.0f, -32.0f, 718.0f }, { 25000, 538.0f, -24.0f, 563.0f }, { 25000, 390.0f, 0.0f, 493.0f }};

    result[Npc::kPotion][Npc::kGrocery] = {{ 25000, 1618.0f, 0.0f, 1078.0f }};
    result[Npc::kPotion][Npc::kBlacksmith] = {{ 25000, 1283.0f, -33.0f, 1299.0f }, { 25000, 667.0f, -32.0f, 1269.0f }, { 25000, 397.0f, 0.0f, 1358.0f }};
    result[Npc::kPotion][Npc::kProtector] = {{ 25000, 1283.0f, -33.0f, 1299.0f }, { 25000, 667.0f, -32.0f, 1269.0f }, { 25000, 363.0f, 0.0f, 1083.0f }};
    result[Npc::kPotion][Npc::kStable] = {{ 25000,  981.0f,  -32.0f,  1032.0f }, { 25000, 390.0f, 0.0f, 493.0f }};

    result[Npc::kBlacksmith][Npc::kProtector] = {{ 25000, 363.0f, 0.0f, 1083.0f }};
    result[Npc::kBlacksmith][Npc::kStable] = {{ 25000, 390.0f, 0.0f, 493.0f }};

    result[Npc::kProtector][Npc::kBlacksmith] = {{ 25000, 397.0f, 0.0f, 1358.0f }};
    result[Npc::kProtector][Npc::kStable] = {{ 25000, 390.0f, 0.0f, 493.0f }};
    
    return result;
  }();

  auto it1 = pathBetween.find(npcSrc);
  if (it1 == pathBetween.end()) {
    LOG() << "No path exists between " << npcSrc << " and " << npcDest << std::endl;
    return { bot_.selfState_.position() };
  }
  auto it2 = it1->second.find(npcDest);
  if (it2 == it1->second.end()) {
    LOG() << "No path exists between " << npcSrc << " and " << npcDest << std::endl;
    return { bot_.selfState_.position() };
  }

  return it2->second;
}

// =====================================================================================================================================
// =====================================================================================================================================
// =====================================================================================================================================

} // namespace state::machine

std::ostream& operator<<(std::ostream &stream, state::machine::Npc npc) {
  switch (npc) {
    case state::machine::Npc::kStorage:
      stream << "Storage";
      break;
    case state::machine::Npc::kPotion:
      stream << "Potion";
      break;
    case state::machine::Npc::kProtector:
      stream << "Protector";
      break;
    case state::machine::Npc::kGrocery:
      stream << "Grocery";
      break;
    case state::machine::Npc::kBlacksmith:
      stream << "Blacksmith";
      break;
    case state::machine::Npc::kStable:
      stream << "Stable";
      break;
    default:
      stream << "UNKNOWN";
      break;
  }
  return stream;
}