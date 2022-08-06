#include "stateMachine.hpp"

#include "bot.hpp"
#include "logging.hpp"
#include "math/position.hpp"
#include "packet/building/clientAgentActionDeselectRequest.hpp";
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentActionTalkRequest.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/clientAgentInventoryStorageOpenRequest.hpp"

namespace state::machine {

// =====================================================================================================================================
// ===============================================================Walking===============================================================
// =====================================================================================================================================

Walking::Walking(const std::vector<packet::structures::Position> &waypoints) : waypoints_(waypoints) {
  if (waypoints_.empty()) {
    throw std::runtime_error("Given empty list of waypoints");
  }
}

void Walking::onUpdate(Bot &bot, const event::Event *event) {
  // TODO: It might be beneficial to have an Event here, if one was triggered
  LOG() << "Walking" << std::endl;
  if (bot.selfState_.moving()) {
    // Still moving, nothing to do
    return;
  }

  // We're not moving
  // Did we just arrive at this waypoint?
  if (math::position::calculateDistance(bot.selfState_.position(), waypoints_[currentWaypointIndex_]) < 5) { // TODO: Choose better measure of "close enough"
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
  bot.broker_.injectPacket(movementPacket, PacketContainer::Direction::kClientToServer);
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

TalkingToStorageNpc::TalkingToStorageNpc() {
  // Figure out what we want to deposit into storage
}

void TalkingToStorageNpc::onUpdate(Bot &bot, const event::Event *event) {
  LOG() << "Talking to storage npc" << std::endl;
  if (npcInteractionState_ == NpcInteractionState::kStart) {
    // Have not yet done anything. First thing is to select the Npc
    LOG() << "Selecting npc" << std::endl;
    const auto selectNpc = packet::building::ClientAgentActionSelectRequest::packet(kStorageNpcGId);
    bot.broker_.injectPacket(selectNpc, PacketContainer::Direction::kClientToServer);
    // Advance state
    npcInteractionState_ = NpcInteractionState::kSelectionRequestPending;
    return;
  }

  if (npcInteractionState_ == NpcInteractionState::kSelectionRequestPending) {
    // TODO: Check if event is selection failed
    if (!bot.selfState_.selectedEntity) {
      // Waiting for npc to be selected, nothing to do
      LOG() << "Waiting until we've selected npc" << std::endl;
      return;
    }

    if (*bot.selfState_.selectedEntity != kStorageNpcGId) {
      throw std::runtime_error("We have something selected, but its not the storage npc");
    }

    // Selection succceeded
    LOG() << "Selected storage npc" << std::endl;
    npcInteractionState_ = NpcInteractionState::kNpcSelected;

    if (bot.selfState_.haveOpenedStorageSinceTeleport) {
      // We have already opened our storage, we just need to talk to the npc
      LOG() << "Storage already opened once, opening shop" << std::endl;
      const auto openStorage = packet::building::ClientAgentActionTalkRequest::packet(*bot.selfState_.selectedEntity, packet::enums::TalkOption::kStorage);
      bot.broker_.injectPacket(openStorage, PacketContainer::Direction::kClientToServer);

      // TODO: We wouldn't have to do this if we could subscribe to our injected packets (via the PacketBroker)
      bot.selfState_.pendingTalkGid = *bot.selfState_.selectedEntity;
      npcInteractionState_ = NpcInteractionState::kShopOpenRequestPending;
    } else {
      // We have not yet opened our storage
      LOG() << "Storage never opened, opening storage first" << std::endl;
      const auto openStorage = packet::building::ClientAgentInventoryStorageOpenRequest::packet(*bot.selfState_.selectedEntity);
      bot.broker_.injectPacket(openStorage, PacketContainer::Direction::kClientToServer);
      // TODO: Once the server responds that this is successful, the client will automatically send the packet for the talk option
      //   To be client-independent, we should block this request from the client and do this ourself
      npcInteractionState_ = NpcInteractionState::kStorageOpenRequestPending;
    }
    return;
  }
  
  if (bot.selfState_.talkingGidAndOption) {
    // We are talking to an npc
    if (bot.selfState_.talkingGidAndOption->first != kStorageNpcGId) {
      throw std::runtime_error("We're talking to some Npc, but it's not the storage Npc");
    }
    if (bot.selfState_.talkingGidAndOption->second != packet::enums::TalkOption::kStorage) {
      throw std::runtime_error("We're talking to the storage Npc, but it's not the storage option");
    }
    npcInteractionState_ = NpcInteractionState::kShopOpened;

    // Storage is open, we're ready to store our items
    storeItems(bot, event);

    if (npcInteractionState_ == NpcInteractionState::kShopOpened) {
      // Still storing items, dont advance
      return;
    }

    // Done storing items
    LOG() << "Done with storage, closing talk dialog" << std::endl;
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(bot.selfState_.talkingGidAndOption->first);
    bot.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
    return;
  }
  
  if (bot.selfState_.selectedEntity) {
    // We have closed the shop, but still have the npc selected. Deselect them
    LOG() << "Done with storage and talk dialog closed, deselecting storage" << std::endl;
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(*bot.selfState_.selectedEntity);
    bot.broker_.injectPacket(packet, PacketContainer::Direction::kClientToServer);
    return;
  }

  if (!bot.selfState_.selectedEntity) {
    // Storage closed and npc deselected, completely done
    LOG() << "Completely done with storage" << std::endl;
    done_ = true;
    return;
  }
}

bool TalkingToStorageNpc::done() const {
  return done_;
}

void TalkingToStorageNpc::storeItems(Bot &bot, const event::Event *event) {
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

      if (bot.selfState_.storage.hasItem(itemToTryStackingSlotNum)) {
        LOG() << "  Slot " << static_cast<int>(itemToTryStackingSlotNum) << " has some item\n";
        // Storage has something in this slot
        auto *item = bot.selfState_.storage.getItem(itemToTryStackingSlotNum);
        const auto *itemExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
        if (itemExpendable != nullptr) {
          // This is a stackable item, lets see if we can stack it somewhere earlier in the storage
          const auto slotsWithThisSameItem = bot.selfState_.storage.findItemsWithRefId(item->refItemId);
          for (auto destSlotNum : slotsWithThisSameItem) {
            if (destSlotNum >= itemToTryStackingSlotNum) {
              // Not going to try to stack this item with itself, or into a later spot
              break;
            }
            auto *destItem = bot.selfState_.storage.getItem(destSlotNum);
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
              const auto moveItemInStoragePacket = packet::building::ClientAgentInventoryOperationRequest::withinStoragePacket(itemToTryStackingSlotNum, destSlotNum, std::min(itemExpendable->quantity, spaceLeftInStack), bot.selfState_.talkingGidAndOption->first);
              bot.broker_.injectPacket(moveItemInStoragePacket, PacketContainer::Direction::kClientToServer);
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
  const auto slotsWithArrows = bot.selfState_.inventory.findItemsWithTypeId(kArrowTypeId);
  const auto slotsWithHpPotions = bot.selfState_.inventory.findItemsWithTypeId(kHpPotionTypeId);
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
    const auto &slot = bot.selfState_.storage.firstFreeSlot();
    if (slot) {
      // Have a free slot in storage
      LOG() << "  Going to move item from slot " << static_cast<int>(slotsOfItemsToStore.front()) << " in inventory to slot " << static_cast<int>(*slot) << " in storage" << std::endl;
      const auto depositItemPacket = packet::building::ClientAgentInventoryOperationRequest::inventoryToStoragePacket(slotsOfItemsToStore.front(), *slot, bot.selfState_.talkingGidAndOption->first);
      bot.broker_.injectPacket(depositItemPacket, PacketContainer::Direction::kClientToServer);
      return;
    }
  }

  // We didnt store anything
  npcInteractionState_ = NpcInteractionState::kDoneStoring;
}
// =====================================================================================================================================
// =============================================================Townlooping=============================================================
// =====================================================================================================================================

TalkingToShopNpc::TalkingToShopNpc(Npc npc) : npc_(npc) {
  LOG() << "Initialized TalkingToShopNpc with npc " << npc_ << std::endl;
}

void TalkingToShopNpc::onUpdate(Bot &bot, const event::Event *event) {
  LOG() << "TalkingToShopNpc::onUpdate with npc " << npc_ << std::endl;
}

bool TalkingToShopNpc::done() const {
  LOG() << "TalkingToShopNpc::done with npc " << npc_ << std::endl;
  return true;
}

// =====================================================================================================================================
// =============================================================Townlooping=============================================================
// =====================================================================================================================================

Townlooping::Townlooping() {
  // Figure out which npcs we want to visit and in what order
  npcsToVisit_ = { Npc::kStorage, Npc::kPotion, Npc::kGrocery, Npc::kBlacksmith, Npc::kProtector, Npc::kStable };
  // Calculate the path to the first Npc
  std::vector<packet::structures::Position> pathToFirstNpc = {
    {25000, 981.0f, -32.0f, 1032.0f}
  };
  // Initialize state as walking
  childState_ = Walking(pathToFirstNpc);
}

void Townlooping::onUpdate(Bot &bot, const event::Event *event) {
TODO_REMOVE_THIS_LABEL:
  if (done()) {
    return;
  }
  LOG() << "Townlooping" << std::endl;
  if (auto *walkingState = std::get_if<Walking>(&childState_)) {
    walkingState->onUpdate(bot, event);

    if (walkingState->done()) {
      // Done walking, advance state
      if (npcsToVisit_[currentNpcIndex_] == Npc::kStorage) {
        LOG() << "Done walking, talking to storage next" << std::endl;
        childState_ = TalkingToStorageNpc();
      } else {
        LOG() << "Done walking, talking to some other npc next" << std::endl;
        childState_ = TalkingToShopNpc(npcsToVisit_[currentNpcIndex_]);
      }
      // TODO: Go back to the top of this function
      goto TODO_REMOVE_THIS_LABEL;
    }
  } else if (auto *talkingToNpcState = std::get_if<TalkingToNpc>(&childState_)) {
    LOG() << "Talking to npc" << std::endl;
    bool doneTalkingToNpc{false};
    if (auto *talkingToStorageNpcState = std::get_if<TalkingToStorageNpc>(talkingToNpcState)) {
      talkingToStorageNpcState->onUpdate(bot, event);
      doneTalkingToNpc = talkingToStorageNpcState->done();
    } else if (auto *talkingToShopNpcState = std::get_if<TalkingToShopNpc>(talkingToNpcState)) {
      talkingToShopNpcState->onUpdate(bot, event);
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
      auto path = pathBetweenNpcs(bot, npcsToVisit_[currentNpcIndex_-1], npcsToVisit_[currentNpcIndex_]);
      // Update our state to walk to the next npc
      LOG() << "Setting our state to  walking" << std::endl;
      childState_ = Walking(path);
      // TODO: Go back to the top of this function
      goto TODO_REMOVE_THIS_LABEL;
    }
  }
}

bool Townlooping::done() const {
  return (currentNpcIndex_ == npcsToVisit_.size());
}

std::vector<packet::structures::Position> Townlooping::pathBetweenNpcs(Bot &bot, Npc npcSrc, Npc npcDest) const {
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
    return { bot.selfState_.position() };
  }
  auto it2 = it1->second.find(npcDest);
  if (it2 == it1->second.end()) {
    LOG() << "No path exists between " << npcSrc << " and " << npcDest << std::endl;
    return { bot.selfState_.position() };
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