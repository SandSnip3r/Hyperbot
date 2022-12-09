#include "talkingToShopNpc.hpp"

#include "bot.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentActionDeselectRequest.hpp"
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentActionTalkRequest.hpp"
#include "packet/building/clientAgentInventoryRepairRequest.hpp"

#include <silkroad_lib/position_math.h>

#include <optional>

namespace state::machine {

TalkingToShopNpc::TalkingToShopNpc(Bot &bot, Npc npc, const std::map<uint32_t, int> &shoppingList) : StateMachine(bot), npc_(npc), shoppingList_(shoppingList) {
  stateMachineCreated(kName);
  // We know we are near our npc, lets find the closest npc to us
  npcGid_ = [&]{
    std::optional<uint32_t> closestNpcGId;
    float closestNpcDistance = std::numeric_limits<float>::max();
    const auto &entityMap = bot_.entityTracker().getEntityMap();
    for (const auto &entityIdObjectPair : entityMap) {
      const auto &entityPtr = entityIdObjectPair.second;
      if (!entityPtr) {
        throw std::runtime_error("Entity map contains a null item");
      }

      if (entityPtr->entityType() != entity::EntityType::kNonplayerCharacter) {
        // Not an npc, skip
        continue;
      }

      const auto distanceToNpc = sro::position_math::calculateDistance2d(bot_.selfState().position(), entityPtr->position());
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

TalkingToShopNpc::~TalkingToShopNpc() {
  stateMachineDestroyed();
}

void TalkingToShopNpc::figureOutWhatToBuy() {
  for (const auto &shoppingItemIdCountPair : shoppingList_) {
    const auto itemRefId = shoppingItemIdCountPair.first;
    // Do we have enough of these in our inventory?
    const auto slotsWithItem = bot_.selfState().inventory.findItemsWithRefId(itemRefId);
    int ownedCountOfItem{0};
    bool haveEnoughOfThisItem{false};
    for (const auto slotWithItem : slotsWithItem) {
      const auto *itemPtr = bot_.selfState().inventory.getItem(slotWithItem);
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

    if (!bot_.gameData().itemData().haveItemWithId(itemRefId)) {
      throw std::runtime_error("Want to buy an item for which we have no data");
    }
    const auto &item = bot_.gameData().itemData().getItemById(itemRefId);
    const auto &nameOfItemToBuy = item.codeName128;
    const auto *npc = bot_.entityTracker().getEntity(npcGid_);
    if (npc == nullptr) {
      throw std::runtime_error("Got entity, but it's null");
    }
    if (npc->entityType() != entity::EntityType::kNonplayerCharacter) {
      throw std::runtime_error("Entity is not a NonplayerCharacter");
    }

    if (!bot_.gameData().characterData().haveCharacterWithId(npc->refObjId)) {
      throw std::runtime_error("Don't have character data for this Npc");
    }
    const auto &character = bot_.gameData().characterData().getCharacterById(npc->refObjId);
    auto &shopTabs = bot_.gameData().shopData().getNpcTabs(character.codeName128);
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
  return buyingItemsChildState_ && buyingItemsChildState_->done();
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
    if (itemAsEquip->repairInvalid(bot_.gameData())) {
      continue;
    }
    if (itemAsEquip->durability < itemAsEquip->maxDurability(bot_.gameData())) {
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

  if (bot_.selfState().talkingGidAndOption) {
    // We are talking to an Npc
    if (bot_.selfState().talkingGidAndOption->first != npcGid_) {
      throw std::runtime_error("We're not talking to the potion Npc that we thought we were");
    }
    if (bot_.selfState().talkingGidAndOption->second != packet::enums::TalkOption::kStore) {
      throw std::runtime_error("We're not in the talk option that we thought we were");
    }
    if (waitingForTalkResponse_) {
      // Successfully began talking to Npc
      waitingForTalkResponse_ = false;
      buyingItemsChildState_ = std::make_unique<BuyingItems>(bot_, itemsToBuy_);
    }

    // Now that we are talking to the npc, start buying items
    if (!buyingItemsChildState_) {
      throw std::runtime_error("If we reach this point, the state must be BuyingItems");
    }
    buyingItemsChildState_->onUpdate(event);
    if (!buyingItemsChildState_->done()) {
      // Still buying items, do not continue
      return;
    }

    // Done buying items at this point
    if (waitingOnStopTalkResponse_) {
      // Already deselected to close the shop, nothing else to do
      return;
    }

    // Close the shop
    const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(bot_.selfState().talkingGidAndOption->first);
    bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
    waitingOnStopTalkResponse_ = true;
    return;
  } else {
    // We are not talking to an npc
    if (bot_.selfState().selectedEntity) {
      // We have something selected
      if (*bot_.selfState().selectedEntity != npcGid_) {
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
        const auto packet = packet::building::ClientAgentActionDeselectRequest::packet(*bot_.selfState().selectedEntity);
        bot_.packetBroker().injectPacket(packet, PacketContainer::Direction::kClientToServer);
        waitingOnDeselectionResponse_ = true;
        return;
      } else {
        // We must talk to the npc
        if (waitingForTalkResponse_) {
          // Already requested talk, nothing to do
          return;
        }

        // Talk to npc
        const auto openStorage = packet::building::ClientAgentActionTalkRequest::packet(*bot_.selfState().selectedEntity, packet::enums::TalkOption::kStore);
        bot_.packetBroker().injectPacket(openStorage, PacketContainer::Direction::kClientToServer);
        waitingForTalkResponse_ = true;
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
        bot_.packetBroker().injectPacket(repairAllPacket, PacketContainer::Direction::kClientToServer);
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
      bot_.packetBroker().injectPacket(selectNpc, PacketContainer::Direction::kClientToServer);
      waitingForSelectionResponse_ = true;
    }
  }
}

bool TalkingToShopNpc::done() const {
  return done_;
}

} // namespace state::machine