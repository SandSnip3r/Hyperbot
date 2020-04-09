#include "skillUseModule.hpp"
#include "../packet/opcode.hpp"
#include "../packet/building/clientAgentActionSelectRequest.hpp"
#include "../packet/building/clientAgentInventoryOperationRequest.hpp"

#include <iostream>
#include <memory>
#include <regex>

namespace module {

SkillUseModule::SkillUseModule(state::Entity &entityState,
                               state::Self &selfState,
                               storage::Storage &inventory,
                               broker::PacketBroker &brokerSystem,
                               broker::EventBroker &eventBroker,
                               const packet::parsing::PacketParser &packetParser,
                               const pk2::GameData &gameData) :
      entityState_(entityState),
      selfState_(selfState),
      inventory_(inventory),
      broker_(brokerSystem),
      eventBroker_(eventBroker),
      packetParser_(packetParser),
      gameData_(gameData) {
  auto packetHandleFunction = std::bind(&SkillUseModule::handlePacket, this, std::placeholders::_1);
  // Client packets
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentChatRequest, packetHandleFunction);
  // Server packets
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionCommandResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionSelectResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentBuffAdd, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentBuffRemove, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateState, packetHandleFunction);

  auto eventHandleFunction = std::bind(&SkillUseModule::handleEvent, this, std::placeholders::_1);
  eventBroker_.subscribeToEvent(event::EventCode::kSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventorySlotUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStatesChanged, eventHandleFunction);
}

bool SkillUseModule::handlePacket(const PacketContainer &packet) {
  std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    std::cerr << "[SkillUseModule] Failed to parse packet " << std::hex << packet.opcode << std::dec << "\n  Error: \"" << ex.what() << "\"\n";
    return true;
  }

  if (!parsedPacket) {
    // Not yet parsing this packet
    return true;
  }

  auto *clientChat = dynamic_cast<packet::parsing::ClientAgentChatRequest*>(parsedPacket.get());
  if (clientChat != nullptr) {
    return clientAgentChatRequestReceived(*clientChat);
  }

  auto *clientAgentBuffAdd = dynamic_cast<packet::parsing::ServerAgentBuffAdd*>(parsedPacket.get());
  if (clientAgentBuffAdd != nullptr) {
    clientAgentBuffAddReceived(*clientAgentBuffAdd);
    return true;
  }

  auto *clientAgentBuffRemove = dynamic_cast<packet::parsing::ServerAgentBuffRemove*>(parsedPacket.get());
  if (clientAgentBuffRemove != nullptr) {
    clientAgentBuffRemoveReceived(*clientAgentBuffRemove);
    return true;
  }

  auto *actionSelectResponse = dynamic_cast<packet::parsing::ServerAgentActionSelectResponse*>(parsedPacket.get());
  if (actionSelectResponse != nullptr) {
    serverAgentActionSelectResponseReceived(*actionSelectResponse);
    return true;
  }

  auto *actionCommandResponse = dynamic_cast<packet::parsing::ServerAgentActionCommandResponse*>(parsedPacket.get());
  if (actionCommandResponse != nullptr) {
    serverAgentActionCommandResponseReceived(*actionCommandResponse);
    return true;
  }

  auto *entityUpdateState = dynamic_cast<packet::parsing::ServerAgentEntityUpdateState*>(parsedPacket.get());
  if (entityUpdateState != nullptr) {
    serverAgentEntityUpdateStateReceived(*entityUpdateState);
    return true;
  }

  std::cout << "SkillUseModule: Unhandled packet subscribed to\n";
  return true;
}

void SkillUseModule::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
  const auto eventCode = event->eventCode;
  switch (eventCode) {
    case event::EventCode::kSpawned: {
      // TODO: Move into some global data?
      activeBuffs_.clear();
      preevaluateBuffs();
      break;
    }
    case event::EventCode::kSkillCooldownEnded: {
      const event::SkillCooldownEnded *skillCooldownEndedEvent = dynamic_cast<const event::SkillCooldownEnded*>(event);
      if (skillCooldownEndedEvent != nullptr) {
        std::cout << "Skill " << skillCooldownEndedEvent->skillRefId << " cooldown ended\n";
        skillsOnCooldown_.erase(skillCooldownEndedEvent->skillRefId);
        castBuffs();
      }
      break;
    }
    case event::EventCode::kInventorySlotUpdated: {
      const event::InventorySlotUpdated *inventorySlotUpdated = dynamic_cast<const event::InventorySlotUpdated*>(event);
      if (inventorySlotUpdated != nullptr) {
        std::cout << "Inventory slot " << (int)inventorySlotUpdated->slotNum << " updated\n";
        if (inventorySlotUpdated->slotNum == 6) {
          weaponChanged();
        } else if (inventorySlotUpdated->slotNum == 7 && inventory_.hasItem(inventorySlotUpdated->slotNum)) {
          std::cout << "Shield equipped, maybe time to use skills\n";
          castBuffs();
        }
      }
      break;
    }
    case event::EventCode::kStatesChanged: {
      std::cout << "Our states changed\n";
      castBuffs();
      break;
    }
  }
}

void SkillUseModule::weaponChanged() {
  if (!tryingToBuff_) {
    return;
  }

  const auto modernStateLevels = selfState_.modernStateLevels();
  if (modernStateLevels[state::toBitNum(packet::enums::AbnormalStateFlag::kStunned)] > 0 ||
      modernStateLevels[state::toBitNum(packet::enums::AbnormalStateFlag::kPetrify)] > 0 ||
      modernStateLevels[state::toBitNum(packet::enums::AbnormalStateFlag::kSleep)] > 0) {
    std::cout << "Stunned, petrified, or sleeping, wont bother trying to change weapons\n";
    return;
  }

  if (inventory_.hasItem(6)) {
    // New weapon in hand
    bool neededToEquipShield = false;
    auto weap = inventory_.getItem(6);
    if (!weap->itemInfo->twoHanded) {
      // Weapon takes a shield
      if (!inventory_.hasItem(7)) {
        // TODO: Use gender to get proper shield
        auto slots = inventory_.findItemsWithTypeId(3,1,4,2);
        if (slots.empty()) {
          std::cout << "Want to equip shield, but not possible\n";
        } else {
          std::cout << "Going to try to equip shield\n";
          broker_.injectPacket(packet::building::ClientAgentInventoryOperationRequest::packet(slots[0], 7, 1), PacketContainer::Direction::kClientToServer);
          neededToEquipShield = true;
        }
      }
    }
    if (!neededToEquipShield) {
      castBuffs();
    }
  }
}

void SkillUseModule::preevaluateBuffs() {
  if (desiredBuffs_.empty()) {
    // Nothing to evaluate
    return;
  }
}
/*
onSpawned {
  auto weap = Find first desired buff that requires a weapon
  equipWeaponForBuff(weap)
}
onState(!untouchable) {
  tryTobuff()
}
onWeaponChanged(weap) {
 if (weap uses shield &&
     shield not equipped) {
    equipShield()
  } else {
    tryTobuff()
  }
}
onShieldEquipped() {
  tryToBuff()
}
onBuffAdded() {
  tryToBuff()
}
onBuffRemoved() {
  tryToBuff()
}
onStatesChanged() {
  tryToBuff
}
onHpMpChanged() {
  tryToBuff()
}
tryTobuff() {
  auto buff = Find first desired buff that is inactive
  if (none) return;
  if (buff.requiredWeapon is not equipped) equipWeaponForBuff(weap)
  else cast(buff)
}
*/

void SkillUseModule::printBuffs() {
  std::cout << "Buffs:\n";
  for (const auto &buff : activeBuffs_) {
    printf(" %6d %6d %s\n",buff.skillRefId, buff.token, gameData_.skillData().getSkillById(buff.skillRefId).basicCode.c_str());
  }
}

void SkillUseModule::clientAgentBuffAddReceived(packet::parsing::ServerAgentBuffAdd &packet) {
  if (packet.globalId() == selfState_.globalId()) {
    std::cout << "Buff " << packet.skillRefId() << " added to us. Active buff token: " << packet.activeBuffToken() << '\n';
    if (packet.activeBuffToken() != 0) {
      // TODO: Better understand token == 0
      activeBuffs_.emplace_back(packet.skillRefId(), packet.activeBuffToken());
    }
  } else {
    std::cout << "Buff " << packet.skillRefId() << " added to " << packet.globalId() << ". Active buff token: " << packet.activeBuffToken() << '\n';
  }
  const auto skill = gameData_.skillData().getSkillById(packet.skillRefId());
  std::cout << "Setting buff cooldown as " << skill.actionReuseDelay << "ms for skill " << packet.skillRefId() << "\n";
  skillsOnCooldown_.emplace(packet.skillRefId());
  eventBroker_.publishDelayedEvent(std::make_unique<event::SkillCooldownEnded>(packet.skillRefId()), std::chrono::milliseconds(skill.actionReuseDelay));
  printBuffs();
  castBuffs();
}

void SkillUseModule::clientAgentBuffRemoveReceived(packet::parsing::ServerAgentBuffRemove &packet) {
  const auto &tokens = packet.tokens();
  std::cout << "Buffs with tokens [ ";
  for (auto i : packet.tokens()) {
    std::cout << i << ", ";
  }
  std::cout << "] removed\n";
  activeBuffs_.erase(std::remove_if(activeBuffs_.begin(), activeBuffs_.end(), [&tokens](const Buff &buff) { return std::find(tokens.begin(), tokens.end(), buff.token) != tokens.end(); }), activeBuffs_.end());
  printBuffs();
  castBuffs();
}

void SkillUseModule::serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) {
  std::cout << "Entity state updated for " << packet.gId() << ", stateType = " << static_cast<uint32_t>(packet.stateType()) << ", state = " << static_cast<uint32_t>(packet.state()) << '\n';
  if (selfState_.spawned() && packet.gId() == selfState_.globalId()) {
    if (packet.stateType() == packet::parsing::StateType::kBodyState) {
      if (static_cast<packet::enums::BodyState>(packet.state()) == packet::enums::BodyState::kNormal) {
        // We're no longer untouchable!
        castBuffs();
      }
    }
  }
  if (packet.stateType() == packet::parsing::StateType::kMotionState) {
    std::cout << "MotionState " << static_cast<uint32_t>(packet.state()) << '\n';
  }
}

void SkillUseModule::serverAgentActionCommandResponseReceived(packet::parsing::ServerAgentActionCommandResponse &packet) {
  if (packet.actionState() == packet::parsing::ActionState::kBegin) {
    // Successfully used skill
  } else if (packet.actionState() == packet::parsing::ActionState::kEnd) {
    // Done using skill
  } else if (packet.actionState() == packet::parsing::ActionState::kError) {
    std::cout << "Error while using skill. Error code :" << packet.errorCode() << "\n";
    // Misused skill, retry? Was it us who cast it?
    // Stunned, maybe dead. 16388
    castBuffs();
  }
}

void SkillUseModule::castBuffs() {
  if (!tryingToBuff_) {
    return;
  }
  std::cout << "Cast buffs\n";

  if (!selfState_.spawned() || (selfState_.lifeState() != packet::enums::LifeState::kAlive && selfState_.lifeState() != packet::enums::LifeState::kEmbryo)) {
    std::cout << "Not spawned/alive, shouldnt bother with buffing\n";
    return;
  }
  
  const auto modernStateLevels = selfState_.modernStateLevels();
  if (modernStateLevels[state::toBitNum(packet::enums::AbnormalStateFlag::kStunned)] > 0 ||
      modernStateLevels[state::toBitNum(packet::enums::AbnormalStateFlag::kPetrify)] > 0 ||
      modernStateLevels[state::toBitNum(packet::enums::AbnormalStateFlag::kSleep)] > 0) {
    std::cout << "Stunned, petrified, or sleeping, wont bother trying to cast buffs\n";
    return;
  }

  std::optional<uint32_t> desiredBuffRefId;
  // Look if there's a buff that we need to cast that uses our current weapon
  if (inventory_.hasItem(6)) {
    auto currentWeapon = inventory_.getItem(6);
    for (auto desiredBuff : desiredBuffs_) {
      if (gameData_.skillData().getSkillById(desiredBuff).reqCastWeapon1 == currentWeapon->itemInfo->typeId4) {
        if (skillsOnCooldown_.find(desiredBuff) == skillsOnCooldown_.end()) {
          if (std::find_if(activeBuffs_.begin(), activeBuffs_.end(), [desiredBuff](const Buff &buff) {
            return buff.skillRefId == desiredBuff;
          }) == activeBuffs_.end()) {
            std::cout << "Buff " << desiredBuff << " is inactive and it requires our current weapon\n";
            desiredBuffRefId = desiredBuff;
            break;
          }
        } else {
          std::cout << "Buff " << desiredBuff << " is on cooldown\n";
        }
      }
    }
  }

  if (!desiredBuffRefId) {
    std::cout << "No buff needs to be cast that uses our current weapon, check all others\n";
    for (auto desiredBuff : desiredBuffs_) {
      if (skillsOnCooldown_.find(desiredBuff) == skillsOnCooldown_.end()) {
        if (std::find_if(activeBuffs_.begin(), activeBuffs_.end(), [desiredBuff](const Buff &buff) {
          return buff.skillRefId == desiredBuff;
        }) == activeBuffs_.end()) {
          desiredBuffRefId = desiredBuff;
          std::cout << "First desired buff is " << *desiredBuffRefId << '\n';
          break;
        }
      } else {
        std::cout << "Buff " << desiredBuff << " is on cooldown\n";
      }
    }
  }

  if (!desiredBuffRefId) {
    std::cout << "Dont have any pending buffs\n";
    return;
  }

  auto reqWeaponTypeId4 = gameData_.skillData().getSkillById(*desiredBuffRefId).reqCastWeapon1;
  bool correctWeaponEquipped = false;
  auto weapItem = inventory_.getItem(6);
  if (weapItem == nullptr) {
    // No weapon equipped
    std::cout << "No weapon equipped\n";
  } else {
    // Some weapon equipped
    std::cout << "Weapon equipped. ";
    if (weapItem->itemInfo->typeId1 == 3 &&
        weapItem->itemInfo->typeId2 == 1 &&
        weapItem->itemInfo->typeId3 == 6 &&
        weapItem->itemInfo->typeId4 == reqWeaponTypeId4) {
      std::cout << "It's a the correct weapon for this skill\n";
      correctWeaponEquipped = true;
    } else {
      std::cout << "But it's not the right weapon for this skill\n";
    }
  }
  // 3 1 6
  // 2  ITEM_CH_SWORD_12_A_RARE
  // 3  ITEM_CH_BLADE_12_A_RARE
  // 4  ITEM_CH_SPEAR_12_A_RARE
  // 5  ITEM_CH_TBLADE_12_A_RARE
  // 6  ITEM_CH_BOW_12_A_RARE
  // 7  ITEM_EU_SWORD_12_A_RARE
  // 8  ITEM_EU_TSWORD_12_A_RARE
  // 9  ITEM_EU_AXE_12_A_RARE
  // 10 ITEM_EU_DARKSTAFF_12_A_RARE
  // 11 ITEM_EU_TSTAFF_12_A_RARE
  // 12 ITEM_EU_CROSSBOW_12_A_RARE
  // 13 ITEM_EU_DAGGER_12_A_RARE
  // 14 ITEM_EU_HARP_12_A_RARE
  // 15 ITEM_EU_STAFF_12_A_RARE

  // TODO: Cast buff that dont require a weapon

  if (!correctWeaponEquipped) {
    std::cout << "Since the incorrect weapon is equipped, equip it\n";
    auto slots = inventory_.findItemsWithTypeId(3,1,6,reqWeaponTypeId4);
    if (!slots.empty()) {
      std::cout << "We found the weapon in slot(s): ";
      for (auto i : slots) {
        std::cout << (int)i << ", ";
      }
      std::cout << '\n';
      std::cout << "Moving item from slot " << (int)slots[0] << " to 6\n";
      broker_.injectPacket(packet::building::ClientAgentInventoryOperationRequest::packet(slots[0], 6, 1), PacketContainer::Direction::kClientToServer);
    } else {
      std::cout << "We couldnt find the weapon\n";
    }
  } else {
    std::cout << "Correct weapon is equipped\n";
    if (selfState_.bodyState() != packet::enums::BodyState::kUntouchable) {
      const auto skill = gameData_.skillData().getSkillById(*desiredBuffRefId);
      if (skill.targetRequired) {
        std::cout << "Casting skill " << (int)*desiredBuffRefId << " on self\n";
        broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::cast(*desiredBuffRefId, selfState_.globalId()), PacketContainer::Direction::kClientToServer);
      } else {
        std::cout << "Casting skill " << (int)*desiredBuffRefId << '\n';
        broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::cast(*desiredBuffRefId), PacketContainer::Direction::kClientToServer);
      }
    } else {
      std::cout << "But we're untouchable, not going to cast\n";
    }
  }
}

void SkillUseModule::selectEntity(state::Entity::EntityId entityId) {
  std::cout << "Being asked to select " << entityId << '\n';
  if (entityState_.trackingEntity(entityId)) {
    auto entity = entityState_.getEntity(entityId);
    std::cout << "Selecting ";
    packet::parsing::printObj(entity, gameData_);
    broker_.injectPacket(packet::building::ClientAgentActionSelectRequest::packet(entityId), PacketContainer::Direction::kClientToServer);
  }
}

void SkillUseModule::commonAttackEntity(state::Entity::EntityId entityId) {
  std::cout << "Being asked to common attack " << entityId << '\n';
  if (entityState_.trackingEntity(entityId)) {
    auto entity = entityState_.getEntity(entityId);
    std::cout << "Common attacking ";
    packet::parsing::printObj(entity, gameData_);
    broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::attack(entityId), PacketContainer::Direction::kClientToServer);
  }
}

void SkillUseModule::traceEntity(state::Entity::EntityId entityId) {
  std::cout << "Being asked to trace " << entityId << '\n';
  if (entityState_.trackingEntity(entityId)) {
    auto entity = entityState_.getEntity(entityId);
    std::cout << "Tracing ";
    packet::parsing::printObj(entity, gameData_);
    broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::trace(entityId), PacketContainer::Direction::kClientToServer);
  }
}

void SkillUseModule::pickupEntity(state::Entity::EntityId entityId) {
  std::cout << "Being asked to pickup " << entityId << '\n';
  if (entityState_.trackingEntity(entityId)) {
    auto entity = entityState_.getEntity(entityId);
    std::cout << "Picking up ";
    packet::parsing::printObj(entity, gameData_);
    broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::pickup(entityId), PacketContainer::Direction::kClientToServer);
  }
}

bool SkillUseModule::clientAgentChatRequestReceived(packet::parsing::ClientAgentChatRequest &packet) {
  std::regex selectGidRegex(R"delim((attack|trace|pickup|select) ([0-9]+))delim");
  std::regex setBuffingRegex(R"delim((buff|pause))delim");
  std::smatch regexMatch;
  if (std::regex_match(packet.message(), regexMatch, selectGidRegex)) {
    const std::string operation = regexMatch[1].str();
    state::Entity::EntityId entityId = std::stoi(regexMatch[2].str());
    if (operation == "select") {
      // selectEntity(entityId);
      castBuffs();
    } else if (operation == "attack") {
      commonAttackEntity(entityId);
    } else if (operation == "trace") {
      traceEntity(entityId);
    } else if (operation == "pickup") {
      pickupEntity(entityId);
    }
    return false;
  } else if (std::regex_match(packet.message(), regexMatch, setBuffingRegex)) {
    const std::string operation = regexMatch[1].str();
    if (operation == "buff") {
      tryingToBuff_ = true;
      castBuffs();
    } else {
      tryingToBuff_ = false;
    }
    return false;
  } else {
    return true;
  }
}

void SkillUseModule::serverAgentActionSelectResponseReceived(packet::parsing::ServerAgentActionSelectResponse &packet) {
  std::cout << "serverAgentActionSelectResponseReceived\n";
  if (packet.result() == 1) {
    // Successfully selected
    std::cout << "Selected successfully\n";
  } else {
    std::cout << "Selection unsuccessful! Error code " << (int)packet.errorCode() << '\n';
  }
}

} // namespace module