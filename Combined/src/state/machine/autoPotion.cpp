#include "autoPotion.hpp"

#include "bot.hpp"
#include "helpers.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "type_id/categories.hpp"

namespace state::machine {

AutoPotion::AutoPotion(Bot &bot) : StateMachine(bot) {
  // TODO:
  //  Rather than constantly searching for them, figure out where every potion and pill is now.
  //  Find HP potions, no grains
  //  Find MP potions, no grains
  //  Find Vigor potions and grains
  //  Find Universal Pills
  //  Find Purification Pills
}

void AutoPotion::onUpdate(const event::Event *event) {
  if (event) {
    // TODO:
    //  Possibly interesting things:
    //   1. Potion/pill moved
    if (const auto *itemReuseDelayEvent = dynamic_cast<const event::ItemWaitForReuseDelay*>(event)) {
      handleItemWaitForReuseDelay(*itemReuseDelayEvent);
    }
  }
  // TODO: Check if we're in a state where using items is possible
  checkIfNeedToUsePill();
  checkIfNeedToHeal();
}

bool AutoPotion::done() const {
  // Autopotion never finishes
  return false;
}

void AutoPotion::checkIfNeedToHeal() {
  if (!bot_.selfState().maxHp() || !bot_.selfState().maxMp()) {
    // Dont yet know our max
    LOG() << "checkIfNeedToHeal: dont know max hp or mp\n";
    return;
  }
  if (*bot_.selfState().maxHp() == 0) {
    // Dead, cant heal
    // TODO: Get from state update instead
    LOG() << "checkIfNeedToHeal: Dead, cant heal\n";
    return;
  }
  const double hpPercentage = static_cast<double>(bot_.selfState().currentHp())/(*bot_.selfState().maxHp());
  const double mpPercentage = static_cast<double>(bot_.selfState().mp())/(*bot_.selfState().maxMp());

  const auto legacyStateEffects = bot_.selfState().legacyStateEffects();
  const bool haveZombie = (legacyStateEffects[helpers::toBitNum(packet::enums::AbnormalStateFlag::kZombie)] > 0);

  // TODO: Investigate if using multiple potions in one go causes issues
  if (!alreadyUsedPotion(PotionType::kVigor)) {
    if (!haveZombie && (hpPercentage < kVigorThreshold_ || mpPercentage < kVigorThreshold_)) {
      usePotion(PotionType::kVigor);
    }
  }
  if (!alreadyUsedPotion(PotionType::kHp)) {
    if (!haveZombie && hpPercentage < kHpThreshold_) {
      usePotion(PotionType::kHp);
    }
  }
  if (!alreadyUsedPotion(PotionType::kMp)) {
    if (mpPercentage < kMpThreshold_) {
      usePotion(PotionType::kMp);
    }
  }
}

bool AutoPotion::alreadyUsedPotion(PotionType potionType) {
  if (potionType == PotionType::kHp) {
    if (bot_.selfState().haveHpPotionEventId()) {
      // On cooldown
      return true;
    }
    const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 1, 1); // TODO: Convert to new type id categories
    return bot_.selfState().itemIsInUsedItemQueue(itemTypeId);
  } else if (potionType == PotionType::kMp) {
    if (bot_.selfState().haveMpPotionEventId()) {
      // On cooldown
      return true;
    }
    const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 1, 2); // TODO: Convert to new type id categories
    return bot_.selfState().itemIsInUsedItemQueue(itemTypeId);
  } else if (potionType == PotionType::kVigor) {
    if (bot_.selfState().haveVigorPotionEventId()) {
      // On cooldown
      return true;
    }
    const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 1, 3); // TODO: Convert to new type id categories
    return bot_.selfState().itemIsInUsedItemQueue(itemTypeId);
  }
  // TODO: Handle other cases
  return false;
}

void AutoPotion::usePotion(PotionType potionType) {
  // We enter this funciton assuming that:
  //  1. The potion isnt on cooldown
  //  2. We have the potion

  uint8_t typeId4;
  if (potionType == PotionType::kHp) {
    typeId4 = 1;
  } else if (potionType == PotionType::kMp) {
    typeId4 = 2;
  } else if (potionType == PotionType::kVigor) {
    typeId4 = 3;
  } else {
    LOG() << "Potion type " << static_cast<int>(potionType) << " not supported\n";
    return;
  }

  // Find potion in inventory
  for (uint8_t slotNum=0; slotNum<bot_.selfState().inventory.size(); ++slotNum) {
    if (!bot_.selfState().inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = bot_.selfState().inventory.getItem(slotNum);
    const storage::ItemExpendable *item;
    if ((item = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 1 && item->itemInfo->typeId4 == typeId4) {
        if (typeId4 == 3 || item->itemInfo->param2 == 0 && item->itemInfo->param4 == 0) {
          // Avoid hp/mp grains
          useItem(slotNum, itemPtr->typeData());
          return;
        }
      }
    }
  }
  // Dont have the item we were looking for
}

void AutoPotion::checkIfNeedToUsePill() {
  const auto legacyStateEffects = bot_.selfState().legacyStateEffects();
  if (std::any_of(legacyStateEffects.begin(), legacyStateEffects.end(), [](const uint16_t effect){ return effect > 0; })) {
    // Need to use a universal pill
    if (!alreadyUsedUniversalPill()) {
      useUniversalPill();
    }
  }
  const auto modernStateLevels = bot_.selfState().modernStateLevels();
  if (std::any_of(modernStateLevels.begin(), modernStateLevels.end(), [](const uint8_t level){ return level > 0; })) {
    // Need to use purification pill
    if (!alreadyUsedPurificationPill()) {
      usePurificationPill();
    }
  }
}

bool AutoPotion::alreadyUsedUniversalPill() {
  if (bot_.selfState().haveUniversalPillEventId()) {
    return true;
  }
  // Pill isnt on cooldown, but maybe we already queued a use of it
  const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 2, 6);
  return bot_.selfState().itemIsInUsedItemQueue(itemTypeId);
}

bool AutoPotion::alreadyUsedPurificationPill() {
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  if (bot_.selfState().havePurificationPillEventId()) {
    return true;
  }
#endif
  const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 2, 1);
  return bot_.selfState().itemIsInUsedItemQueue(itemTypeId);
}

void AutoPotion::useUniversalPill() {
  // Figure out our status with the highest effect
  const auto legacyStateEffects = bot_.selfState().legacyStateEffects();
  uint16_t ourWorstStatusEffect = *std::max_element(legacyStateEffects.begin(), legacyStateEffects.end());
  int32_t bestCure = 0;
  uint8_t bestOptionSlotNum;
  type_id::TypeId bestOptionTypeData;

  for (uint8_t slotNum=0; slotNum<bot_.selfState().inventory.size(); ++slotNum) {
    if (!bot_.selfState().inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = bot_.selfState().inventory.getItem(slotNum);
    const storage::ItemExpendable *item;
    if ((item = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 2 && item->itemInfo->typeId4 == 6) {
        // Universal pill
        if (bestCure == 0) {
          // First pill found, at least we can use this
          bestCure = item->itemInfo->param1;
          bestOptionSlotNum = slotNum;
          bestOptionTypeData = itemPtr->typeData();
        } else {
          // Already have a choice, lets see if this is better
          const auto thisPillCureEffect = item->itemInfo->param1;
          const bool curesEverything = (thisPillCureEffect >= ourWorstStatusEffect);
          const bool curesMoreThanPrevious = (thisPillCureEffect >= ourWorstStatusEffect && bestCure < ourWorstStatusEffect);
          if (curesEverything && thisPillCureEffect < bestCure) {
            // Found a smaller pill that can cure everything
            bestCure = thisPillCureEffect;
            bestOptionSlotNum = slotNum;
            bestOptionTypeData = itemPtr->typeData();
          } else if (curesMoreThanPrevious && thisPillCureEffect > bestCure) {
            // Found a pill that can cure more without being wasteful
            bestCure = thisPillCureEffect;
            bestOptionSlotNum = slotNum;
            bestOptionTypeData = itemPtr->typeData();
          }
        }
      }
    }
  }
  if (bestCure != 0) {
    // Found the best pill
    useItem(bestOptionSlotNum, bestOptionTypeData);
  }
}

void AutoPotion::usePurificationPill() {
  const auto modernStateLevels = bot_.selfState().modernStateLevels();
  int32_t currentCureLevel = 0;
  uint8_t bestOptionSlotNum;
  type_id::TypeId bestOptionTypeData;

  for (uint8_t slotNum=0; slotNum<bot_.selfState().inventory.size(); ++slotNum) {
    if (!bot_.selfState().inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = bot_.selfState().inventory.getItem(slotNum);
    const storage::ItemExpendable *item;
    if ((item = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 2 && item->itemInfo->typeId4 == 1) {
        // Purification pill
        const auto pillCureStateBitmask = item->itemInfo->param1;
        const auto curableStatesWeHave = (pillCureStateBitmask & bot_.selfState().stateBitmask());
        if (curableStatesWeHave > 0) {
          // This pill will cure at least some of the type of state(s) that we have
          const auto pillTreatmentLevel = item->itemInfo->param2;
          if (pillTreatmentLevel != currentCureLevel) {
            std::vector<uint8_t> stateLevels;
            for (uint32_t bitNum=0; bitNum<32; ++bitNum) {
              const auto bit = 1 << bitNum;
              if (curableStatesWeHave & bit) {
                stateLevels.push_back(modernStateLevels[bitNum]);
              }
            }
            const bool curesEverything = (*std::max_element(stateLevels.begin(), stateLevels.end()) <= pillTreatmentLevel);
            const bool curesMoreThanPrevious = (std::find_if(stateLevels.begin(), stateLevels.end(), [&pillTreatmentLevel, &currentCureLevel](const uint8_t lvl){
              return ((lvl > currentCureLevel) && (lvl <= pillTreatmentLevel));
            }) != stateLevels.end());

            if (pillTreatmentLevel < currentCureLevel && curesEverything) {
              // Found a smaller pill that is completely sufficient
              currentCureLevel = pillTreatmentLevel;
              bestOptionSlotNum = slotNum;
              bestOptionTypeData = itemPtr->typeData();
            } else if (pillTreatmentLevel > currentCureLevel && curesMoreThanPrevious) {
              // Found a bigger pill that does more than the previous
              currentCureLevel = pillTreatmentLevel;
              bestOptionSlotNum = slotNum;
              bestOptionTypeData = itemPtr->typeData();
            }
          }
        }
      }
    }
  }
  if (currentCureLevel != 0) {
    // Found the best pill
    useItem(bestOptionSlotNum, bestOptionTypeData);
  }
}

void AutoPotion::useItem(uint8_t slotNum, type_id::TypeId typeData) {
  if (type_id::categories::kRecoveryPotion.contains(typeData)) {
    if (type_id::categories::kHpPotion.contains(typeData)) {
      if (alreadyUsedPotion(PotionType::kHp)) {
        // Already used an Hp potion, not going to re-queue
        return;
      }
    } else if (type_id::categories::kMpPotion.contains(typeData)) {
      if (alreadyUsedPotion(PotionType::kMp)) {
        // Already used an Mp potion, not going to re-queue
        return;
      }
    } else if (type_id::categories::kVigorPotion.contains(typeData)) {
      if (alreadyUsedPotion(PotionType::kVigor)) {
        // Already used a Vigor potion, not going to re-queue
        return;
      }
    }
  }
  bot_.packetBroker().injectPacket(packet::building::ClientAgentInventoryItemUseRequest::packet(slotNum, typeData), PacketContainer::Direction::kClientToServer);
  // TODO: Refactor everything below, maybe instead, this should go into a client packet handler
  bot_.selfState().pushItemToUsedItemQueue(slotNum, typeData);
  if (bot_.selfState().itemUsedTimeoutTimer) {
    LOG() << "WARNING: Already have a running timer for item use" << std::endl;
  }
  bot_.selfState().itemUsedTimeoutTimer = bot_.eventBroker().publishDelayedEvent(std::make_unique<event::ItemUseTimeout>(slotNum, typeData), std::chrono::milliseconds(100)); // TODO: What timeout should we use? Probably something related to server ping
}

void AutoPotion::handleItemWaitForReuseDelay(const event::ItemWaitForReuseDelay &event) {
  LOG() << "Failed to use ";
  if (type_id::categories::kHpPotion.contains(event.itemTypeId)) {
    std::cout << "hp";
  } else if (type_id::categories::kMpPotion.contains(event.itemTypeId)) {
    std::cout << "mp";
  } else if (type_id::categories::kVigorPotion.contains(event.itemTypeId)) {
    std::cout << "vigor";
  } else {
    std::cout << "unknown item" << std::endl;
    return;
  }
  std::cout << " potion because there's still a cooldown, going to retry" << std::endl;
  useItem(event.inventorySlotNum, event.itemTypeId);
}

} // namespace state::machine