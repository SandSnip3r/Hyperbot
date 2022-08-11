#include "stateMachine.hpp"

#include "bot.hpp"
#include "logging.hpp"
#include "math/position.hpp"
#include "packet/building/clientAgentActionDeselectRequest.hpp"
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentActionTalkRequest.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/clientAgentInventoryStorageOpenRequest.hpp"
#include "packet/building/serverAgentInventoryOperationResponse.hpp"

namespace state::machine {

CommonStateMachine::CommonStateMachine(Bot &bot) : bot_(bot) {
}

void CommonStateMachine::blockOpcode(packet::Opcode opcode) {
  if (bot_.proxy_.blockingOpcode(opcode)) {
    LOG() << "Someone is already blocking opcode " << packet::toStr(opcode) << ". Not going to block" << std::endl;
  } else {
    LOG() << "Blocking opcode " << packet::toStr(opcode) << std::endl;
    bot_.proxy_.blockOpcode(opcode);
    blockedOpcodes_.push_back(opcode);
  }
}

CommonStateMachine::~CommonStateMachine() {
  // Undo all blocked opcodes
  for (const auto opcode : blockedOpcodes_) {
    LOG() << "Unblocking opcode " << packet::toStr(opcode) << std::endl;
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
  // TODO: It might be beneficial to have an Event here, if one was triggered
  LOG() << "Walking" << std::endl;
  if (bot_.selfState_.moving()) {
    // Still moving, nothing to do
    return;
  }

  // We're not moving
  // Did we just arrive at this waypoint?
  if (math::position::calculateDistance(bot_.selfState_.position(), waypoints_[currentWaypointIndex_]) < 5) { // TODO: Choose better measure of "close enough"
    // Just arrived at the waypoint, increment index
    LOG() << "Arrived at waypoint" << std::endl;
    ++currentWaypointIndex_;
    requestedMovement_ = false;
  } else if (requestedMovement_) {
    // Already asked to move, nothing to do
    LOG() << "Waiting on pending movement request" << std::endl;
    return;
  }

  // We are not moving, and we do not have a pending movement request
  if (done()) {
    // Finished walking
    LOG() << "Done walking" << std::endl;
    return;
  }

  // We are not moving, we're not at the current waypoint, and there's not a pending movement request
  // Send a request to move to the current waypoint
  LOG() << "Moving to waypoint\n";
  const auto &currentWaypoint = waypoints_[currentWaypointIndex_];
  const auto movementPacket = packet::building::ClientAgentCharacterMoveRequest::packet(currentWaypoint.regionId, currentWaypoint.xOffset, currentWaypoint.yOffset, currentWaypoint.zOffset);
  bot_.broker_.injectPacket(movementPacket, PacketContainer::Direction::kClientToServer);
  requestedMovement_ = true;
}

bool Walking::done() const {
  return (currentWaypointIndex_ == waypoints_.size());
}

// =====================================================================================================================================
// =========================================================TalkingToStorageNpc=========================================================
// =====================================================================================================================================

const uint16_t TalkingToStorageNpc::kArrowTypeId = helpers::type_id::makeTypeId(3,3,4,1);
const uint16_t TalkingToStorageNpc::kHpPotionTypeId = helpers::type_id::makeTypeId(3,3,1,1);

TalkingToStorageNpc::TalkingToStorageNpc(Bot &bot) : bot_(bot) {
  // Figure out what we want to deposit into storage
}

void TalkingToStorageNpc::onUpdate(const event::Event *event) {
  LOG() << "Talking to storage npc" << std::endl;
  if (npcInteractionState_ == NpcInteractionState::kStart) {
    // Have not yet done anything. First thing is to select the Npc
    LOG() << "Selecting npc" << std::endl;
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
      LOG() << "Waiting until we've selected npc" << std::endl;
      return;
    }

    if (*bot_.selfState_.selectedEntity != kStorageNpcGId) {
      throw std::runtime_error("We have something selected, but its not the storage npc");
    }

    // Selection succceeded
    LOG() << "Selected storage npc" << std::endl;
    npcInteractionState_ = NpcInteractionState::kNpcSelected;

    if (bot_.selfState_.haveOpenedStorageSinceTeleport) {
      // We have already opened our storage, we just need to talk to the npc
      LOG() << "Storage already opened once, opening shop" << std::endl;
      const auto openStorage = packet::building::ClientAgentActionTalkRequest::packet(*bot_.selfState_.selectedEntity, packet::enums::TalkOption::kStorage);
      bot_.broker_.injectPacket(openStorage, PacketContainer::Direction::kClientToServer);

      // TODO: We wouldn't have to do this if we could subscribe to our injected packets (via the PacketBroker)
      bot_.selfState_.pendingTalkGid = *bot_.selfState_.selectedEntity;
      npcInteractionState_ = NpcInteractionState::kShopOpenRequestPending;
    } else {
      // We have not yet opened our storage
      LOG() << "Storage never opened, opening storage first" << std::endl;
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
    LOG() << "Done with storage, closing talk dialog" << std::endl;
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(bot_.selfState_.talkingGidAndOption->first);
    bot_.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
    return;
  } else if (npcInteractionState_ == NpcInteractionState::kShopOpenRequestPending || npcInteractionState_ == NpcInteractionState::kStorageOpenRequestPending) {
    LOG() << "Still waiting to talk to storage" << std::endl;
    return;
  }
  
  if (bot_.selfState_.selectedEntity) {
    // We have closed the shop, but still have the npc selected. Deselect them
    LOG() << "Done with storage and talk dialog closed, deselecting storage" << std::endl;
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(*bot_.selfState_.selectedEntity);
    bot_.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
    return;
  }

  if (!bot_.selfState_.selectedEntity) {
    // Storage closed and npc deselected, completely done
    LOG() << "Completely done with storage" << std::endl;
    done_ = true;
    return;
  }
}

bool TalkingToStorageNpc::done() const {
  return done_;
}

void TalkingToStorageNpc::storeItems(const event::Event *event) {
  // Did something just arrive in storage?
  LOG() << "Storing items" << std::endl;
  if (event != nullptr) {
    if (dynamic_cast<const event::InventoryUpdated*>(event) != nullptr) {
      // Ignoring inventory updated events since a corresponding storage updated packet will come with more info for us
      return;
    }
    const auto *storageUpdatedEvent = dynamic_cast<const event::StorageUpdated*>(event);
    if (storageUpdatedEvent != nullptr) {
      LOG() << "  Storage updated event\n";
      // At least one storage slot was updated
      pendingItemMovementRequest_ = false;
      uint8_t itemToTryStackingSlotNum;
      if (storageUpdatedEvent->destSlotNum.has_value()) {
        // Something was moved in storage. Either a deposit or an item stacked in storage
        if (storageUpdatedEvent->srcSlotNum.has_value()) {
          // This was a move within storage, a stacking in this case
          // The src slot could still contain an item, we'll try to stack that
          LOG() << "  This was a move within storage, a stacking in this case" << std::endl;
          itemToTryStackingSlotNum = *storageUpdatedEvent->srcSlotNum;
        } else {
          // This was a deposit, try to stack the newly added item
          LOG() << "  This was a deposit" << std::endl;
          itemToTryStackingSlotNum = *storageUpdatedEvent->destSlotNum;
        }
      }

      if (bot_.selfState_.storage.hasItem(itemToTryStackingSlotNum)) {
        LOG() << "  Slot " << static_cast<int>(itemToTryStackingSlotNum) << " has some item\n";
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
            LOG() << "  Slot " << static_cast<int>(destSlotNum) << " has our item and currently has a stack size of " << spaceLeftInStack << std::endl;
            if (spaceLeftInStack > 0) {
              // We can stack our item to this spot
              LOG() << "  Going to stack item in storage from slot " << static_cast<int>(itemToTryStackingSlotNum) << " to slot " << static_cast<int>(destSlotNum) << ". Src quantity: " << itemExpendable->quantity << ", dest quantity: " << destItemExpendable->quantity << std::endl;
              const auto moveItemInStoragePacket = packet::building::ClientAgentInventoryOperationRequest::withinStoragePacket(itemToTryStackingSlotNum, destSlotNum, std::min(itemExpendable->quantity, spaceLeftInStack), bot_.selfState_.talkingGidAndOption->first);
              bot_.broker_.injectPacket(moveItemInStoragePacket, PacketContainer::Direction::kClientToServer);
              // TODO: There is a potential that this function gets retriggered before the item movement completes. Maybe we ought to set an internal state to block that from messing us up
              pendingItemMovementRequest_ = true;
              return;
            }
          }
          LOG() << "  Didnt find a place to stack this item to\n";
        }
      }
    }
  }

  if (pendingItemMovementRequest_) {
    // Still waiting for an item to move, nothing to do
    return;
  }

  LOG() << "  Nothing to stack, try storing items" << std::endl;
  // At this point, we werent able to stack, move the next item
  // What should we store?
  const auto slotsWithArrows = bot_.selfState_.inventory.findItemsWithTypeId(kArrowTypeId);
  const auto slotsWithHpPotions = bot_.selfState_.inventory.findItemsWithTypeId(kHpPotionTypeId);
  auto slotsOfItemsToStore = slotsWithArrows;
  slotsOfItemsToStore.insert(slotsOfItemsToStore.end(), slotsWithHpPotions.begin(), slotsWithHpPotions.end());
  std::sort(slotsOfItemsToStore.begin(), slotsOfItemsToStore.end());
  if (!slotsOfItemsToStore.empty()) {
    LOG() << "  We have items to store: ";
    for (auto i : slotsOfItemsToStore) {
      std::cout << static_cast<int>(i) << ' ';
    }
    std::cout << std::endl;
    // Try to store first item
    // Figure out where to store it
    const auto &slot = bot_.selfState_.storage.firstFreeSlot();
    if (slot) {
      // Have a free slot in storage
      LOG() << "  Going to move item from slot " << static_cast<int>(slotsOfItemsToStore.front()) << " in inventory to slot " << static_cast<int>(*slot) << " in storage" << std::endl;
      const auto depositItemPacket = packet::building::ClientAgentInventoryOperationRequest::inventoryToStoragePacket(slotsOfItemsToStore.front(), *slot, bot_.selfState_.talkingGidAndOption->first);
      bot_.broker_.injectPacket(depositItemPacket, PacketContainer::Direction::kClientToServer);
      return;
    }
  }

  // We didnt store anything
  npcInteractionState_ = NpcInteractionState::kDoneStoring;
}
// =====================================================================================================================================
// =============================================================Townlooping=============================================================
// =====================================================================================================================================

TalkingToShopNpc::TalkingToShopNpc(Bot &bot, Npc npc) : CommonStateMachine(bot), npc_(npc) {
  LOG() << "Initialized TalkingToShopNpc with npc " << npc_ << std::endl;
  if (npc_ == Npc::kPotion) {
    npcGid_ = 0x0111;
    // Figure out what items to items to buy
    constexpr const uint32_t kXLargeHpPotionRefId{8};
    constexpr const int kXLargeHpPotionBuyCount{1};
    itemsToBuy_[kXLargeHpPotionRefId] = kXLargeHpPotionBuyCount;

    blockOpcode(packet::Opcode::kServerAgentActionSelectResponse);
    blockOpcode(packet::Opcode::kServerAgentActionTalkResponse);
    blockOpcode(packet::Opcode::kServerAgentInventoryOperationResponse);
  }
}

void TalkingToShopNpc::onUpdate(const event::Event *event) {
  LOG() << "TalkingToShopNpc::onUpdate with npc " << npc_ << std::endl;
  if (npc_ == Npc::kPotion) {
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
      }
      LOG() << "We are talking to the potion Npc. Ready to buy items" << std::endl;
      buyItems(event);

      if (!doneBuyingItems_) {
        // Still buying items, do not continue
        LOG() << "Still buying items, do not continue" << std::endl;
        return;
      }

      if (waitingOnStopTalkResponse_) {
        // Already deselected to close the shop, nothing else to do
        LOG() << "Already deselected to close the shop, nothing else to do" << std::endl;
        return;
      }

      LOG() << "Must be done buying items, probably time to close the shop" << std::endl;
      const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(bot_.selfState_.talkingGidAndOption->first);
      bot_.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
      waitingOnStopTalkResponse_ = true;
      return;
    } else {
      // We are not talking to an npc
      if (bot_.selfState_.selectedEntity) {
        // We have something selected
        if (*bot_.selfState_.selectedEntity != npcGid_) {
          throw std::runtime_error("We have something selected, but it's not the potion Npc");
        }
        if (waitingForSelectionResponse_) {
          LOG() << "Successfully selected potion Npc" << std::endl;
          waitingForSelectionResponse_ = false;
        }
        // We're either about to buy items, or just finished
        if (doneBuyingItems_) {
          // We must deselect the npc
          if (waitingOnDeselectionResponse_) {
            // Already deselected, nothing to do
            LOG() << "Already deselected, nothing to do" << std::endl;
            return;
          }

          LOG() << "We must deselect the npc" << std::endl;
          const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(*bot_.selfState_.selectedEntity);
          bot_.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
          waitingOnDeselectionResponse_ = true;
          return;
        } else {
          // We must talk to the npc
          LOG() << "We must talk to the npc" << std::endl;
          if (waitingForTalkResponse_) {
            // Already requested talk, nothing to do
            return;
          }
          const auto openStorage = packet::building::ClientAgentActionTalkRequest::packet(*bot_.selfState_.selectedEntity, packet::enums::TalkOption::kStore);
          bot_.broker_.injectPacket(openStorage, PacketContainer::Direction::kClientToServer);
          waitingForTalkResponse_ = true;

          // TODO: We wouldn't have to do this if we could subscribe to our injected packets (via the PacketBroker)
          bot_.selfState_.pendingTalkGid = *bot_.selfState_.selectedEntity;
        }
      } else {
        // No Npc is selected
        if (doneBuyingItems_) {
          done_ = true;
          return;
        }
        if (waitingForSelectionResponse_) {
          // Already requested selection, nothing to do
          return;
        }
        LOG() << "No Npc is selected, selecting" << std::endl;
        const auto selectNpc = packet::building::ClientAgentActionSelectRequest::packet(npcGid_);
        bot_.broker_.injectPacket(selectNpc, PacketContainer::Direction::kClientToServer);
        waitingForSelectionResponse_ = true;
      }
    }
  }
}

bool TalkingToShopNpc::done() const {
  if (npc_ == Npc::kPotion) {
    if (done_) {
      LOG() << "Done with potion npc!" << std::endl;
    }
    return done_;
  }

  LOG() << "TalkingToShopNpc::done with npc " << npc_ << std::endl;
  return true;
}

void TalkingToShopNpc::buyItems(const event::Event *event) {
  LOG() << "Buying items" << std::endl;

  if (event) {
    if (auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event)) {
      if (inventoryUpdatedEvent->destSlotNum) {
        // TODO: We dont actually know if this was our purchase, for now, we assume it was
        if (inventoryUpdatedEvent->srcSlotNum) {
          // This was a stacking, or item movement. This packet should actually be forwarded to the client, but we block it
          // TODO: Handle. Ideally fix the packet forwarding mechanism. A temporary fix would be to just rebuild the packet and inject it (that might not be possible since we dont know how many items were in each stack before the stacking)
          throw std::runtime_error("Not currently handling stacking of items while buying from an Npc");
        }
        LOG() << "Inventory slot " << static_cast<int>(*inventoryUpdatedEvent->destSlotNum) << " has received an item" << std::endl;
        const auto *itemAtInventorySlot = bot_.selfState_.inventory.getItem(*inventoryUpdatedEvent->destSlotNum);
        if (itemAtInventorySlot == nullptr) {
          throw std::runtime_error("Got an item from our inventory, but there's nothing here");
        }
        auto it = itemsToBuy_.find(itemAtInventorySlot->refItemId);
        if (it == itemsToBuy_.end()) {
          throw std::runtime_error("Thought we bought an item, but its not in our to-buy list");
        }
        const auto beforeCount = it->second;
        if (const auto *itemExp = dynamic_cast<const storage::ItemExpendable*>(itemAtInventorySlot)) {
          it->second -= itemExp->quantity;
        } else {
          --(it->second);
        }
        LOG() << "Reducing remaining buy count from " << beforeCount << " to " << it->second << std::endl;
        if (it->second < 0) {
          throw std::runtime_error("Somehow bought more than we wanted");
        } else if (it->second == 0) {
          // No more of these to buy, delete from shopping list
          itemsToBuy_.erase(it);
        }
        waitingOnBuyResponse_ = false;

        // The packet which tells the client about this purchase has been blocked. We need to spoof an item spawning in the character's inventory
        const auto itemBuySpoofPacket = packet::building::ServerAgentInventoryOperationResponse::addItemByServerPacket(*inventoryUpdatedEvent->destSlotNum, *itemAtInventorySlot);
        bot_.broker_.injectPacket(itemBuySpoofPacket, PacketContainer::Direction::kServerToClient);
      }
    }
  }

  if (waitingOnBuyResponse_) {
    // Waiting on an item we bought, nothing to do
    return;
  }

  if (itemsToBuy_.empty()) {
    // Nothing else to buy
    LOG() << "Nothing else to buy" << std::endl;
    doneBuyingItems_ = true;
    return;
  }

  struct PurchaseRequest {
    uint8_t tabIndex;
    uint8_t itemIndex;
    uint16_t quantity;
  };
  const auto nextPurchaseRequest = [&]() -> PurchaseRequest {
    auto nextItemToBuyIt = itemsToBuy_.begin();
    if (nextItemToBuyIt->second <= 0) {
      throw std::runtime_error("Non-positive number of items to buy");
    }
    // Get name of item
    if (!bot_.gameData_.itemData().haveItemWithId(nextItemToBuyIt->first)) {
      throw std::runtime_error("Want to buy an item for which we have no data");
    }
    const auto &item = bot_.gameData_.itemData().getItemById(nextItemToBuyIt->first);
    const auto &nameOfItemToBuy = item.codeName128;
    // Try to find item in shop

    const auto *object = bot_.entityState_.getEntity(*bot_.selfState_.selectedEntity);
    const auto *npc = dynamic_cast<const packet::parsing::NonplayerCharacter*>(object);
    if (npc == nullptr) {
      throw std::runtime_error("Npc is not a NonplayerCharacter");
    }

    LOG() << " Npc's ref id is " << npc->refObjId << std::endl;
    if (!bot_.gameData_.characterData().haveCharacterWithId(npc->refObjId)) {
      throw std::runtime_error("Dont have character data");
    }
    const auto &character = bot_.gameData_.characterData().getCharacterById(npc->refObjId);
    LOG() << " This npc's name is " << character.codeName128 << std::endl;
    auto &shopTabs = bot_.gameData_.shopData().getNpcTabs(character.codeName128);
    LOG() << " This npc has " << shopTabs.size() << " tabs" << std::endl;
    for (int tabIndex=0; tabIndex<shopTabs.size(); ++tabIndex) {
      const auto &tab = shopTabs[tabIndex];
      const auto &packageMap = tab.getPackageMap();
      LOG() << "  Tab name \"" << tab.getName() << "\" has " << packageMap.size() << " item(s)" << std::endl;
      for (const auto &itemIndexAndScrapPair : packageMap) {
        LOG() << "   Slot " << static_cast<int>(itemIndexAndScrapPair.first) << " has item " << itemIndexAndScrapPair.second.refItemCodeName << std::endl;
        if (itemIndexAndScrapPair.second.refItemCodeName == nameOfItemToBuy) {
          LOG() << "    This is the item we want to buy!" << std::endl;
          return { static_cast<uint8_t>(tabIndex), itemIndexAndScrapPair.first, static_cast<uint16_t>(std::min(item.maxStack, nextItemToBuyIt->second)) };
        }
      }
    }
  }();
  LOG() << "We want to purchase item on tab " << static_cast<int>(nextPurchaseRequest.tabIndex) << ", with item index " << static_cast<int>(nextPurchaseRequest.itemIndex) << " and quantity " << nextPurchaseRequest.quantity << std::endl;
  const auto buyItemPacket = packet::building::ClientAgentInventoryOperationRequest::buyPacket(nextPurchaseRequest.tabIndex, nextPurchaseRequest.itemIndex, nextPurchaseRequest.quantity, bot_.selfState_.talkingGidAndOption->first);
  bot_.broker_.injectPacket(buyItemPacket, PacketContainer::Direction::kClientToServer);
  {
    // TODO: We wouldn't have to do this if we could subscribe to our injected packets (via the PacketBroker)
    packet::structures::ItemMovement itemMovement;
    itemMovement.type = packet::enums::ItemMovementType::kBuyFromNPC;
    itemMovement.globalId = bot_.selfState_.talkingGidAndOption->first;
    itemMovement.storeTabNumber = nextPurchaseRequest.tabIndex;
    itemMovement.storeSlotNumber = nextPurchaseRequest.itemIndex;
    bot_.selfState_.setUserPurchaseRequest(itemMovement);
  }
  waitingOnBuyResponse_ = true;
}

// =====================================================================================================================================
// =============================================================Townlooping=============================================================
// =====================================================================================================================================

Townlooping::Townlooping(Bot &bot) : CommonStateMachine(bot) {
  // Block packets while we're in the townlooping state

  // Figure out which npcs we want to visit and in what order
  npcsToVisit_ = { Npc::kStorage, Npc::kPotion/* , Npc::kGrocery, Npc::kBlacksmith, Npc::kProtector, Npc::kStable */ };
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
  LOG() << "Townlooping" << std::endl;
  if (auto *walkingState = std::get_if<Walking>(&childState_)) {
    walkingState->onUpdate(event);

    if (walkingState->done()) {
      // Done walking, advance state
      childState_.emplace<TalkingToNpc>();
      auto &talkingToNpcState = std::get<TalkingToNpc>(childState_);
      if (npcsToVisit_[currentNpcIndex_] == Npc::kStorage) {
        LOG() << "Done walking, talking to storage next" << std::endl;
        talkingToNpcState.emplace<TalkingToStorageNpc>(bot_);
      } else {
        LOG() << "Done walking, talking to some other npc next" << std::endl;
        talkingToNpcState.emplace<TalkingToShopNpc>(bot_, npcsToVisit_[currentNpcIndex_]);
        LOG() << "Done constructing" << std::endl;
      }
      // TODO: Go back to the top of this function
      goto TODO_REMOVE_THIS_LABEL;
    }
  } else if (auto *talkingToNpcState = std::get_if<TalkingToNpc>(&childState_)) {
    LOG() << "Talking to npc" << std::endl;
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
      LOG() << "Done talking to npc" << std::endl;
      ++currentNpcIndex_;
      if (done()) {
        // No more Npcs, done with townloop
        LOG() << "No more npcs to visit, done with townloop" << std::endl;
        return;
      }

      // Calculate the path from the just-finished npc to the next npc
      auto path = pathBetweenNpcs(npcsToVisit_[currentNpcIndex_-1], npcsToVisit_[currentNpcIndex_]);
      // Update our state to walk to the next npc
      LOG() << "Setting our state to  walking" << std::endl;
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