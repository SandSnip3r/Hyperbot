#include "bot.hpp"
#include "helpers.hpp"
#include "logging.hpp"

#include "packet/building/clientAgentAuthRequest.hpp"
#include "packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "packet/building/clientGatewayLoginIbuvAnswer.hpp"
#include "packet/building/clientGatewayLoginRequest.hpp"

Bot::Bot(const config::CharacterLoginData &loginData,
         const pk2::GameData &gameData,
         broker::PacketBroker &broker) :
      loginData_(loginData),
      gameData_(gameData),
      broker_(broker) {
  eventBroker_.run();
  userInterface_.run();
  subscribeToEvents();
}

void Bot::subscribeToEvents() {
  auto eventHandleFunction = std::bind(&Bot::handleEvent, this, std::placeholders::_1);
  // Login events
  eventBroker_.subscribeToEvent(event::EventCode::kStateShardIdUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateConnectedToAgentServerUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateReceivedCaptchaPromptUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateCharacterListUpdated, eventHandleFunction);
  // Movement events
  eventBroker_.subscribeToEvent(event::EventCode::kMovementEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterSpeedUpdated, eventHandleFunction);
  // Character info events
  eventBroker_.subscribeToEvent(event::EventCode::kItemWaitForReuseDelay, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kHpPotionCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpPotionCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kVigorPotionCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kUniversalPillCooldownEnded, eventHandleFunction);
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  eventBroker_.subscribeToEvent(event::EventCode::kPurificationPillCooldownEnded, eventHandleFunction);
#endif
  eventBroker_.subscribeToEvent(event::EventCode::kHpPercentChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpPercentChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStatesChanged, eventHandleFunction);
}

void Bot::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);

  const auto eventCode = event->eventCode;
  switch (eventCode) {
    // Login events
    case event::EventCode::kStateShardIdUpdated:
      handleStateShardIdUpdated();
      break;
    case event::EventCode::kStateConnectedToAgentServerUpdated:
      handleStateConnectedToAgentServerUpdated();
      break;
    case event::EventCode::kStateReceivedCaptchaPromptUpdated:
      handleStateReceivedCaptchaPromptUpdated();
      break;
    case event::EventCode::kStateCharacterListUpdated:
      handleStateCharacterListUpdated();
      break;
    // Movement events
    case event::EventCode::kMovementEnded:
      handleMovementEnded();
      break;
    case event::EventCode::kCharacterSpeedUpdated:
      handleSpeedUpdated();
      break;
    // Character info events
    case event::EventCode::kItemWaitForReuseDelay:
      {
        const event::ItemWaitForReuseDelay &castedEvent = dynamic_cast<const event::ItemWaitForReuseDelay&>(*event);
        handleItemWaitForReuseDelay(castedEvent);
      }
      break;
    case event::EventCode::kHpPotionCooldownEnded:
    case event::EventCode::kMpPotionCooldownEnded:
    case event::EventCode::kVigorPotionCooldownEnded:
      handlePotionCooldownEnded(eventCode);
      break;
    case event::EventCode::kUniversalPillCooldownEnded:
    case event::EventCode::kPurificationPillCooldownEnded:
      handlePillCooldownEnded(eventCode);
      break;
    case event::EventCode::kHpPercentChanged:
    case event::EventCode::kMpPercentChanged:
      handleVitalsChanged();
      break;
    case event::EventCode::kStatesChanged:
      handleStatesChanged();
      break;
    default:
      LOG() << "Unhandled event subscribed to. Code:" << static_cast<int>(eventCode) << '\n';
      break;
  }
}

// ============================================================================================================================
// ================================================Login process event handling================================================
// ============================================================================================================================

void Bot::handleStateShardIdUpdated() const {
  // We received the server list from the server, try to log in
  const auto loginAuthPacket = packet::building::ClientGatewayLoginRequest::packet(gameData_.divisionInfo().locale, loginData_.id, loginData_.password, selfState_.shardId);
  broker_.injectPacket(loginAuthPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateConnectedToAgentServerUpdated() {
  const auto clientAuthPacket = packet::building::ClientAgentAuthRequest::packet(selfState_.token, loginData_.id, loginData_.password, gameData_.divisionInfo().locale, selfState_.kMacAddress);
  broker_.injectPacket(clientAuthPacket, PacketContainer::Direction::kClientToServer);
  // Set our state to logging in so that we'll know to block packets from the client if it tries to also login
  selfState_.loggingIn = true;
}

void Bot::handleStateReceivedCaptchaPromptUpdated() const {
  LOG() << "Got captcha. Sending answer\n";
  const auto captchaAnswerPacket = packet::building::ClientGatewayLoginIbuvAnswer::packet(selfState_.kCaptchaAnswer);
  broker_.injectPacket(captchaAnswerPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateCharacterListUpdated() const {
  LOG() << "Char list received: [ ";
  for (const auto &i : selfState_.characterList) {
    std::cout << i.name << ' ';
  }
  std::cout << "]\n";

  // Search for our character in the character list
  auto it = std::find_if(selfState_.characterList.begin(), selfState_.characterList.end(), [this](const packet::structures::CharacterSelection::Character &character) {
    return character.name == loginData_.name;
  });
  if (it == selfState_.characterList.end()) {
    LOG() << "Unable to find character \"" << loginData_.name << "\"\n";
    return;
  }

  // Found our character, select it
  auto charSelectionPacket = packet::building::ClientAgentCharacterSelectionJoinRequest::packet(loginData_.name);
  broker_.injectPacket(charSelectionPacket, PacketContainer::Direction::kClientToServer);
}

// ============================================================================================================================
// ==================================================Movement event handling===================================================
// ============================================================================================================================

void Bot::handleSpeedUpdated() {
  if (selfState_.haveMovingEventId() && selfState_.moving()) {
    if (selfState_.haveDestination()) {
      // Need to update timer
      auto seconds = helpers::secondsToTravel(selfState_.position(), selfState_.destination(), selfState_.currentSpeed());
      eventBroker_.cancelDelayedEvent(selfState_.getMovingEventId());
      const auto movingEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
      selfState_.setMovingEventId(movingEventId);
    }
  }
}

void Bot::handleMovementEnded() {
  selfState_.resetMovingEventId();
  LOG() << "Movement ended event\n";
  selfState_.doneMoving();
  LOG() << "Currently at " << selfState_.position() << '\n';
}

// ============================================================================================================================
// ===============================================Character info packet handling===============================================
// ============================================================================================================================

void Bot::handleItemWaitForReuseDelay(const event::ItemWaitForReuseDelay &castedEvent) {
  LOG() << "Failed to use ";
  if (castedEvent.itemTypeId == helpers::type_id::makeTypeId(3, 3, 1, 1)) {
    std::cout << "hp";
  } else if (castedEvent.itemTypeId == helpers::type_id::makeTypeId(3, 3, 1, 2)) {
    std::cout << "mp";
  } else if (castedEvent.itemTypeId == helpers::type_id::makeTypeId(3, 3, 1, 3)) {
    std::cout << "vigor";
  }
  std::cout << " potion because there's still a cooldown, going to retry" << std::endl;
  useItem(castedEvent.inventorySlotNum, castedEvent.itemTypeId);
}

void Bot::handlePotionCooldownEnded(const event::EventCode eventCode) {
  LOG() << "Potion cooldown ended" << std::endl;
  if (eventCode == event::EventCode::kHpPotionCooldownEnded) {
    selfState_.resetHpPotionEventId();
    checkIfNeedToHeal(); // TODO: Instead, retrigger the game loop
  } else if (eventCode == event::EventCode::kMpPotionCooldownEnded) {
    selfState_.resetMpPotionEventId();
    checkIfNeedToHeal(); // TODO: Instead, retrigger the game loop
  } else if (eventCode == event::EventCode::kVigorPotionCooldownEnded) {
    selfState_.resetVigorPotionEventId();
    checkIfNeedToHeal(); // TODO: Instead, retrigger the game loop
  }
}

void Bot::handlePillCooldownEnded(const event::EventCode eventCode) {
  LOG() << "Pill cooldown ended" << std::endl;
  if (eventCode == event::EventCode::kUniversalPillCooldownEnded) {
    selfState_.resetUniversalPillEventId();
    checkIfNeedToUsePill(); // TODO: Instead, retrigger the game loop
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  } else if (eventCode == event::EventCode::kPurificationPillCooldownEnded) {
    selfState_.resetPurificationPillEventId();
    checkIfNeedToUsePill(); // TODO: Instead, retrigger the game loop
#endif
  }
}

void Bot::handleVitalsChanged() {
  checkIfNeedToHeal();

  // Broadcast message to UI
  if (!selfState_.maxHp() || !selfState_.maxMp()) {
    // Dont yet know our max
    std::cout << "handleVitalsChanged: dont know max hp or mp\n";
    return;
  }

  broadcast::HpMpUpdate hpMpUpdate;
  hpMpUpdate.set_currenthp(selfState_.hp());
  hpMpUpdate.set_maxhp(*selfState_.maxHp());
  hpMpUpdate.set_currentmp(selfState_.mp());
  hpMpUpdate.set_maxmp(*selfState_.maxMp());
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_hpmpupdate() = hpMpUpdate;
  userInterface_.broadcast(broadcastMessage);
}

void Bot::handleStatesChanged() {
  checkIfNeedToUsePill();
}

// ============================================================================================================================
// ====================================================Actual action logic=====================================================
// ============================================================================================================================

void Bot::checkIfNeedToHeal() {
  if (!selfState_.maxHp() || !selfState_.maxMp()) {
    // Dont yet know our max
    LOG() << "checkIfNeedToHeal: dont know max hp or mp\n";
    return;
  }
  if (*selfState_.maxHp() == 0) {
    // Dead, cant heal
    // TODO: Get from state update instead
    LOG() << "checkIfNeedToHeal: Dead, cant heal\n";
    return;
  }
  const double hpPercentage = static_cast<double>(selfState_.hp())/(*selfState_.maxHp());
  const double mpPercentage = static_cast<double>(selfState_.mp())/(*selfState_.maxMp());

  const auto legacyStateEffects = selfState_.legacyStateEffects();
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

bool Bot::alreadyUsedPotion(PotionType potionType) {
  if (potionType == PotionType::kHp) {
    if (selfState_.haveHpPotionEventId()) {
      return true;
    }
    const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 1, 1);
    return selfState_.itemIsInUsedItemQueue(itemTypeId);
  } else if (potionType == PotionType::kMp) {
    if (selfState_.haveMpPotionEventId()) {
      return true;
    }
    const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 1, 2);
    return selfState_.itemIsInUsedItemQueue(itemTypeId);
  } else if (potionType == PotionType::kVigor) {
    if (selfState_.haveVigorPotionEventId()) {
      return true;
    }
    const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 1, 3);
    return selfState_.itemIsInUsedItemQueue(itemTypeId);
  }
  // TODO: Handle other cases
  return false;
}

void Bot::usePotion(PotionType potionType) {
  // We enter this funciton assuming that:
  //  1. The potion isnt on cooldown
  //  2. We have the potion
  const double hpPercentage = static_cast<double>(selfState_.hp())/(*selfState_.maxHp()); // TODO: Remove, for print only
  const double mpPercentage = static_cast<double>(selfState_.mp())/(*selfState_.maxMp()); // TODO: Remove, for print only
  printf("Healing. Hp: %4.2f%%, Mp: %4.2f%%\n", hpPercentage*100, mpPercentage*100); std::cout.flush();

  uint8_t typeId4;
  if (potionType == PotionType::kHp) {
    typeId4 = 1;
  } else if (potionType == PotionType::kMp) {
    typeId4 = 2;
  } else if (potionType == PotionType::kVigor) {
    typeId4 = 3;
  } else {
    std::cout << "CharacterInfoModule::usePotion: Potion type " << static_cast<int>(potionType) << " not supported\n";
    return;
  }

  // Find potion in inventory
  for (uint8_t slotNum=0; slotNum<selfState_.inventory.size(); ++slotNum) {
    if (!selfState_.inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = selfState_.inventory.getItem(slotNum);
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

void Bot::checkIfNeedToUsePill() {
  const auto legacyStateEffects = selfState_.legacyStateEffects();
  if (std::any_of(legacyStateEffects.begin(), legacyStateEffects.end(), [](const uint16_t effect){ return effect > 0; })) {
    // Need to use a universal pill
    if (!alreadyUsedUniversalPill()) {
      useUniversalPill();
    }
  }
  const auto modernStateLevels = selfState_.modernStateLevels();
  if (std::any_of(modernStateLevels.begin(), modernStateLevels.end(), [](const uint8_t level){ return level > 0; })) {
    // Need to use purification pill
    if (!alreadyUsedPurificationPill()) {
      usePurificationPill();
    }
  }
}

bool Bot::alreadyUsedUniversalPill() {
  if (selfState_.haveUniversalPillEventId()) {
    return true;
  }
  // Pill isnt on cooldown, but maybe we already queued a use of it
  const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 2, 6);
  return selfState_.itemIsInUsedItemQueue(itemTypeId);
}

bool Bot::alreadyUsedPurificationPill() {
#ifdef ENFORCE_PURIFICATION_PILL_COOLDOWN
  if (selfState_.purificationPillOnCooldown()) {
    return true;
  }
#endif
  const auto itemTypeId = helpers::type_id::makeTypeId(3, 3, 2, 1);
  return selfState_.itemIsInUsedItemQueue(itemTypeId);
}

void Bot::useUniversalPill() {
  // Figure out our status with the highest effect
  const auto legacyStateEffects = selfState_.legacyStateEffects();
  uint16_t ourWorstStatusEffect = *std::max_element(legacyStateEffects.begin(), legacyStateEffects.end());
  int32_t bestCure = 0;
  uint8_t bestOptionSlotNum;
  uint16_t bestOptionTypeData;

  for (uint8_t slotNum=0; slotNum<selfState_.inventory.size(); ++slotNum) {
    if (!selfState_.inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = selfState_.inventory.getItem(slotNum);
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

void Bot::usePurificationPill() {
  const auto modernStateLevels = selfState_.modernStateLevels();
  int32_t currentCureLevel = 0;
  uint8_t bestOptionSlotNum;
  uint16_t bestOptionTypeData;

  for (uint8_t slotNum=0; slotNum<selfState_.inventory.size(); ++slotNum) {
    if (!selfState_.inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *itemPtr = selfState_.inventory.getItem(slotNum);
    const storage::ItemExpendable *item;
    if ((item = dynamic_cast<const storage::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 2 && item->itemInfo->typeId4 == 1) {
        // Purification pill
        const auto pillCureStateBitmask = item->itemInfo->param1;
        const auto curableStatesWeHave = (pillCureStateBitmask & selfState_.stateBitmask());
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

void Bot::useItem(uint8_t slotNum, uint16_t typeData) {
  uint8_t typeId1 = (typeData >> 2) & 0b111;
  uint8_t typeId2 = (typeData >> 5) & 0b11;
  uint8_t typeId3 = (typeData >> 7) & 0b1111;
  uint8_t typeId4 = (typeData >> 11) & 0b11111;
  if (typeId1 == 3 && typeId2 == 3 && typeId3 == 1) {
    // Potion
    if (typeId4 == 1) {
      if (alreadyUsedPotion(PotionType::kHp)) {
        // Already used an Hp potion, not going to re-queue
        return;
      }
    } else if (typeId4 == 2) {
      if (alreadyUsedPotion(PotionType::kMp)) {
        // Already used an Mp potion, not going to re-queue
        return;
      }
    } else if (typeId4 == 3) {
      if (alreadyUsedPotion(PotionType::kVigor)) {
        // Already used a Vigor potion, not going to re-queue
        return;
      }
    }
  }
  broker_.injectPacket(packet::building::ClientAgentInventoryItemUseRequest::packet(slotNum, typeData), PacketContainer::Direction::kClientToServer);
  selfState_.pushItemToUsedItemQueue(slotNum, typeData);
}