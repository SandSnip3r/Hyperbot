#include "skillUseModule.hpp"
#include "../packet/opcode.hpp"
#include "../packet/building/clientAgentActionSelectRequest.hpp"
#include "../packet/building/clientAgentInventoryOperationRequest.hpp"
#include "../packet/building/serverAgentChatUpdate.hpp"

#include <iostream>
#include <memory>
#include <regex>

#define LOG(level) printf("[%13lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());\
                   std::cout << "["#level"] "

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
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentActionCommandRequest, packetHandleFunction);
  // Server packets
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionCommandResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionSelectResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentBuffAdd, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentBuffRemove, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateState, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillBegin, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillEnd, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentChatUpdate, packetHandleFunction);

  auto eventHandleFunction = std::bind(&SkillUseModule::handleEvent, this, std::placeholders::_1);
  eventBroker_.subscribeToEvent(event::EventCode::kSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventorySlotUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStatesChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kHpPercentChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpPercentChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillCastAboutToEnd, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockbackStatusEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kTemp, eventHandleFunction);
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

  auto *commandRequest = dynamic_cast<packet::parsing::ClientAgentActionCommandRequest*>(parsedPacket.get());
  if (commandRequest != nullptr) {
    clientAgentActionCommandRequestReceived(*commandRequest);
    return true;
  }

  auto *buffAdd = dynamic_cast<packet::parsing::ServerAgentBuffAdd*>(parsedPacket.get());
  if (buffAdd != nullptr) {
    serverAgentBuffAddReceived(*buffAdd);
    return true;
  }

  auto *buffRemove = dynamic_cast<packet::parsing::ServerAgentBuffRemove*>(parsedPacket.get());
  if (buffRemove != nullptr) {
    serverAgentBuffRemoveReceived(*buffRemove);
    return true;
  }

  auto *selectResponse = dynamic_cast<packet::parsing::ServerAgentActionSelectResponse*>(parsedPacket.get());
  if (selectResponse != nullptr) {
    serverAgentActionSelectResponseReceived(*selectResponse);
    return true;
  }

  auto *commandResponse = dynamic_cast<packet::parsing::ServerAgentActionCommandResponse*>(parsedPacket.get());
  if (commandResponse != nullptr) {
    serverAgentActionCommandResponseReceived(*commandResponse);
    return true;
  }

  auto *entityUpdateState = dynamic_cast<packet::parsing::ServerAgentEntityUpdateState*>(parsedPacket.get());
  if (entityUpdateState != nullptr) {
    serverAgentEntityUpdateStateReceived(*entityUpdateState);
    return true;
  }

  auto *skillBegin = dynamic_cast<packet::parsing::ServerAgentSkillBegin*>(parsedPacket.get());
  if (skillBegin != nullptr) {
    serverAgentSkillBeginReceived(*skillBegin);
    return true;
  }

  auto *skillEnd = dynamic_cast<packet::parsing::ServerAgentSkillEnd*>(parsedPacket.get());
  if (skillEnd != nullptr) {
    serverAgentSkillEndReceived(*skillEnd);
    return true;
  }

  auto *serverChat = dynamic_cast<packet::parsing::ServerAgentChatUpdate*>(parsedPacket.get());
  if (serverChat != nullptr) {
    serverAgentChatUpdateReceived(*serverChat);
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
      std::cout << "EVENT: Spawned\n";
      activeBuffs_.clear();
      break;
    }
    case event::EventCode::kSkillCooldownEnded: {
      const event::SkillCooldownEnded *skillCooldownEndedEvent = dynamic_cast<const event::SkillCooldownEnded*>(event);
      if (skillCooldownEndedEvent != nullptr) {
        printf("[%13lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
        std::cout << "  Skill " << skillCooldownEndedEvent->skillRefId << " cooldown ended\n";
        skillsOnCooldown_.erase(skillCooldownEndedEvent->skillRefId);
      }
      break;
    }
    case event::EventCode::kInventorySlotUpdated: {
      const event::InventorySlotUpdated *inventorySlotUpdated = dynamic_cast<const event::InventorySlotUpdated*>(event);
      if (inventorySlotUpdated != nullptr) {
        if (inventorySlotUpdated->slotNum == 6) {
          std::cout << "EVENT: Weapon moved ";
          if (inventory_.hasItem(6)) {
            std::cout << "(equipped or changed)\n";
          } else {
            std::cout << "(removed)\n";
          }
        } else if (inventorySlotUpdated->slotNum == 7 && inventory_.hasItem(inventorySlotUpdated->slotNum)) {
          // Shield equipped, maybe time to use skills
          // TODO: Arrows can trigger this event
          std::cout << "EVENT: Shield equipped\n";
        }
      }
      break;
    }
    case event::EventCode::kHpPercentChanged: {
      float ratio =  selfState_.hp() / static_cast<float>(selfState_.maxHp().value_or(1));
      if (healthIsSafe_) {
        if (ratio < kHpSafetyThreshold_) {
          std::cout << "EVENT: Hp fell below threshold, "<< ratio*100 << "%, " << selfState_.hp() << '/' << selfState_.maxHp().value_or(0) << '\n';
          healthIsSafe_ = false;
        }
      } else {
        if (ratio >= kHpSafetyThreshold_) {
          std::cout << "EVENT: Hp is above threshold, "<< ratio*100 << "%, " << selfState_.hp() << '/' << selfState_.maxHp().value_or(0) << '\n';
          healthIsSafe_ = true;
        }
      }
      break;
    }
    case event::EventCode::kMpPercentChanged:
      // std::cout << "EVENT: Mp Changed, " << selfState_.mp() << '/' << selfState_.maxMp().value_or(0) << '\n';
      break;
    case event::EventCode::kStatesChanged: {
      if (!(stateBitmask_ & static_cast<uint32_t>(packet::enums::AbnormalStateFlag::kStunned)) &&
          (selfState_.stateBitmask() & static_cast<uint32_t>(packet::enums::AbnormalStateFlag::kStunned))) {
        // Just got stunned
        std::cout << "Just got stunned!\n";
        // Looks like B074-End comes for us
        // Dont expect B071 for current skill(s)        
        waitingForCast_ = false;
      } else if ((stateBitmask_ & static_cast<uint32_t>(packet::enums::AbnormalStateFlag::kStunned)) &&
                 !(selfState_.stateBitmask() & static_cast<uint32_t>(packet::enums::AbnormalStateFlag::kStunned))) {
        // Stun just expired
        std::cout << "Stun ended!\n";
      }
      stateBitmask_ = selfState_.stateBitmask();
      break;
    }
    case event::EventCode::kSkillCastAboutToEnd: {
      LOG(Info) << "  Waiting for skill cast completed\n";
      waitingForCast_ = false;
      break;
    }
    case event::EventCode::kKnockbackStatusEnded: {
      printf("[%13lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
      std::cout << "  Knockback status ended\n";
      knockedBackCooldownEventId_.reset();
      break;
    }
    case event::EventCode::kTemp:
      tryCastNext();
      break;
  }
}

void SkillUseModule::serverAgentChatUpdateReceived(packet::parsing::ServerAgentChatUpdate &packet) {
  std::regex startPvpRegex(R"delim(go)delim");
  std::smatch regexMatch;
  const auto str = packet.message();
  if (std::regex_match(str, regexMatch, startPvpRegex)) {
    if (packet.chatType() == packet::enums::ChatType::kAll ||
        packet.chatType() == packet::enums::ChatType::kAllGm) {
      std::cout << "Going to try to attack " << packet.senderGlobalId() << '\n';
      targetGId_ = packet.senderGlobalId();
      attackMode_ = true;
      attack();
    } else if (packet.chatType() == packet::enums::ChatType::kPm) {
      // TODO: Search character entities by name, get Global ID
    }
  }
}

bool SkillUseModule::skillInQueue(uint32_t skillId) {
  if (activeAction_) {
    if (activeAction_->actionType == packet::enums::ActionType::kCast) {
      if (activeAction_->refSkillId == skillId) {
        return true;
      }
    }
  }
  if (queuedAction_) {
    if (queuedAction_->actionType == packet::enums::ActionType::kCast) {
      if (queuedAction_->refSkillId == skillId) {
        return true;
      }
    }
  }
  if (queuedInstantSkills_.find(skillId) != queuedInstantSkills_.end()) {
    return true;
  }
  return std::find_if(pendingActionQueue_.begin(), pendingActionQueue_.end(), [&skillId](const packet::structures::ActionCommand &action){
    if (action.actionType == packet::enums::ActionType::kCast) {
      if (action.refSkillId == skillId) {
        return true;
      }
    }
    return false;
  }) != pendingActionQueue_.end();
}

packet::structures::ActionCommand createActionCommandCastSkill(uint32_t skillId, std::optional<int32_t> targetGlobalId = {}) {
  packet::structures::ActionCommand action;
  action.commandType = packet::enums::CommandType::kExecute;
  action.actionType = packet::enums::ActionType::kCast;
  action.refSkillId = skillId;
  if (targetGlobalId) {
    action.targetType = packet::enums::TargetType::kEntity;
  } else {
    action.targetType = packet::enums::TargetType::kNone;
  }
  return action;
}

void SkillUseModule::attack() {
  if (!attackMode_) {
    return;
  }
  if (selfState_.lifeState() != packet::enums::LifeState::kAlive &&
      selfState_.lifeState() != packet::enums::LifeState::kEmbryo) {
    std::cout << "Cant attack because not alive\n";
    return;
  }
  if (selfState_.bodyState() == packet::enums::BodyState::kUntouchable) {
    std::cout << "Cant attack because untouchable\n";
    return;
  }
  if (selfState_.stateBitmask() & static_cast<uint32_t>(packet::enums::AbnormalStateFlag::kStunned)) {
    // Unable to attack
    std::cout << "Cant attack because stunned\n";
    return;
  }
  if (knockedBackCooldownEventId_) {
    std::cout << "Cant attack because recently knocked back\n";
    return;
  }
  std::cout << "-> Attack!\n";
  bool nonInstantSkillAlreadyInQueue = (std::find_if(pendingActionQueue_.begin(), pendingActionQueue_.end(), [this](const packet::structures::ActionCommand &action) {
    if (action.actionType == packet::enums::ActionType::kCast) {
      return !gameData_.skillData().getSkillById(action.refSkillId).isInstant();
    }
    return false;
  }) != pendingActionQueue_.end());
  // Assume skill in prioritized order
  for (const auto &skillId : skillsToUse_) {
    const auto &skill = gameData_.skillData().getSkillById(skillId);
    if ((skill.basicActivity == 1) ||
        (!waitingForCast_ &&
         !queuedAction_ &&
         !nonInstantSkillAlreadyInQueue)) {
      // TODO: This should better check pendingActions, like pickup, dispel, etc.
      // Skill can be cast while another is being cast or
      //  not currently casting another and
      //  have space in the skill queue and
      //  there is no non-instant skill in the pending queue   
      if (skillsOnCooldown_.find(skillId) == skillsOnCooldown_.end()) {
        // Skill not on cooldown
        if (!skillInQueue(skillId)) {
          // Skill not already in queue
          // If its not a buff that we already have active
          if (std::find_if(activeBuffs_.begin(), activeBuffs_.end(), [&skillId](const Buff &buff){
            return buff.skillRefId == skillId;
          }) == activeBuffs_.end()) {
            if (skill.actionOverlap == 1) {
              // Skill is a chinese imbue
              // TODO: Expand for rogue poison imbue
              if (activeAction_) {
                const auto activeSkill = gameData_.skillData().getSkillById(activeAction_->refSkillId);
                if (activeAction_->actionType == packet::enums::ActionType::kAttack ||
                    activeSkill.targetRequired && (activeSkill.targetGroupEnemy_M || activeSkill.targetGroupEnemy_P)) {
                  // Using an attack skill
                  packet::structures::ActionCommand action;
                  broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::cast(skillId), PacketContainer::Direction::kClientToServer);
                  action = createActionCommandCastSkill(skillId);
                  printf("[%13lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
                  std::cout << ">>>>> Using an attack skill, imbuing " << skillId << "\n";
                  requestedAction(action);
                  break;
                }
              }
            } else {
              packet::structures::ActionCommand action;
              if (skill.targetRequired && (skill.targetGroupEnemy_M || skill.targetGroupEnemy_P)) {
                // An attack skill
                broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::cast(skillId, targetGId_), PacketContainer::Direction::kClientToServer);
                action = createActionCommandCastSkill(skillId, targetGId_);
              } else {
                broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::cast(skillId), PacketContainer::Direction::kClientToServer);
                action = createActionCommandCastSkill(skillId);
              }
              printf("[%13lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
              std::cout << ">>>>> Trying to use skill " << skillId << "\n";
              requestedAction(action);
              break;
            }
          }
        }
      }
    }
  }
}

void SkillUseModule::targetDied() {
  // std::optional<packet::structures::ActionCommand> activeAction_, queuedAction_;
  std::cout << "Target died\n";
  std::cout << "  Deactivating attack mode\n";
  attackMode_ = false;
  std::cout << "  Clearing next skill queue\n";
  queuedAction_.reset();
  pendingActionQueue_.clear();
  printQueues();
}

void SkillUseModule::died() {
  std::cout << "We died\n";
  std::cout << "  Deactivating attack mode\n";
  attackMode_ = false;
  std::cout << "  Clearing next skill queue\n";
  activeAction_.reset();
  queuedAction_.reset();
  pendingActionQueue_.clear();
  waitingForCast_ = false;
  knockedBackCooldownEventId_.reset();
  printQueues();
}

void SkillUseModule::knockedBack() {
  std::cout << "We just got knocked back!\n";
  activeAction_.reset();
  queuedAction_.reset();
  pendingActionQueue_.clear();
  waitingForCast_ = false;
  printQueues();
  knockedBackCooldownEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kKnockbackStatusEnded), std::chrono::milliseconds(kKnockbackStatusDurationMs));
}

void SkillUseModule::knockedDown() {
  std::cout << "We just got knocked down!\n";
  activeAction_.reset();
  queuedAction_.reset();
  pendingActionQueue_.clear();
  waitingForCast_ = false;
  printQueues();
  // Seems knockdown doesnt have a formal delay
  // knockedBackCooldownEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kKnockbackStatusEnded), std::chrono::milliseconds(kKnockbackStatusDurationMs));
}

bool isCommonAttack(const pk2::ref::Skill &skill) {
  // TODO: Improve
  // Punch not included
  return (skill.id == 70 || skill.id == 40 || skill.id == 2 || skill.id == 8421 || skill.id == 9354 || skill.id == 9355 || skill.id == 11162 || skill.id == 9944 || skill.id == 8419 || skill.id == 8420 || skill.id == 11526 || skill.id == 10625);
}

void print(const packet::structures::ActionCommand &action) {
  using namespace packet::enums;
  std::cout << '{';
  if (action.commandType == CommandType::kExecute) {
    if (action.actionType == ActionType::kAttack) {
      std::cout << "Attack," << action.targetGlobalId;
    } else if (action.actionType == ActionType::kPickup) {
      std::cout << "Pickup";
    } else if (action.actionType == ActionType::kTrace) {
      std::cout << "Trace";
    } else if (action.actionType == ActionType::kDispel) {
      std::cout << "Dispel " << action.refSkillId;
    } else if (action.actionType == ActionType::kCast) {
      std::cout << "Cast " << action.refSkillId;
      if (action.targetType == packet::enums::TargetType::kEntity) {
        std::cout << ',' << action.targetGlobalId;
      }
    }
  } else {
    std::cout << "Cancel";
  }
  std::cout << '}';
}

// =====================================================================================================================
// =====================================================[%5d] BEGIN=====================================================
// =====================================================================================================================
void SkillUseModule::serverAgentSkillBeginReceived(packet::parsing::ServerAgentSkillBegin &packet) {
  auto saveCastAndPrint = [this](uint32_t castId, uint32_t refSkillId) {
    this->castSkillMap_.emplace(castId, refSkillId);
    LOG(Info) << "  Cast skill map [ ";
    for (const auto castSkill : this->castSkillMap_) {
      std::cout << '(' << castSkill.first << ',' << castSkill.second << "), ";
    }
    std::cout << "]\n";
  };
  if (packet.result() == 1) {
    LOG(Info);
    printf("[%5d] BEGIN %6d\n", packet.castId(), packet.refSkillId());
    for (const auto &hitObject : packet.action().hitObjects) {
      for (const auto &hit : hitObject.hits) {
        if (hit.hitResult == packet::enums::HitResult::kKill &&
            hitObject.objGlobalId == targetGId_) {
          LOG(Info) << "  Target died " << (int)hit.damageFlag << ' ' << hit.damage << "\n";
          // Our target died
          targetDied();
        } else if ((hitObject.objGlobalId == selfState_.globalId()) && hit.hitResult == packet::enums::HitResult::kKnockback) {
          knockedBack();
        } else if ((hitObject.objGlobalId == selfState_.globalId()) && hit.hitResult == packet::enums::HitResult::kKnockdown) {
          // knockedDown();
        }
      }
    }
    if (packet.casterGlobalId() == selfState_.globalId()) {
      const auto skill = gameData_.skillData().getSkillById(packet.refSkillId());
      if (isCommonAttack(skill)) {
        // Common attack
        LOG(Info) << "  Common attack. Target " << packet.targetGlobalId() << '\n';
        if (queuedCommonAttack_) {
          if (activeAction_) {
            if (activeAction_->commandType == packet::enums::CommandType::kExecute &&
                activeAction_->actionType == packet::enums::ActionType::kAttack) {
              LOG(Warning) << "  Already common attacking but have a queued common attack?\n";
            } else {
              LOG(Warning) << "  Have an active action (";
              print(*activeAction_);
              std::cout << ") while transitioning to common attacking\n";
              // Shouldnt be possible. AC END should have cleared this
            }
          } else {
            LOG(Info) << "  Begin common attacking\n";
          }
          if (queuedCommonAttack_->targetGlobalId != packet.targetGlobalId()) {
            LOG(Error) << "  Queued common attack and our current common attack have different targets\n";
          }
          activeAction_ = queuedCommonAttack_;
          queuedCommonAttack_.reset();
        } else if (activeAction_->commandType != packet::enums::CommandType::kExecute ||
                   activeAction_->actionType != packet::enums::ActionType::kAttack) {
          LOG(Info) << "  Common attack, but no queued common attack and not currently common attacking\n";
          // Create common attack as active action
          activeAction_.emplace();
          activeAction_->commandType = packet::enums::CommandType::kExecute;
          activeAction_->actionType = packet::enums::ActionType::kAttack;
          activeAction_->targetType = packet::enums::TargetType::kEntity;
          activeAction_->targetGlobalId = packet.targetGlobalId();
          LOG(Info) << "  Creating active action as common attack\n";
        }
        if (queuedAction_) {
          LOG(Info) << "  Destroying queued action\n";
          queuedAction_.reset();
        }
      } else {
        // Set skill cooldown
        if (!activeChainSkillId_) {
          LOG(Info) << "  Setting skill " << packet.refSkillId() << " cooldown as " << skill.actionReuseDelay << "ms\n";
          eventBroker_.publishDelayedEvent(std::make_unique<event::SkillCooldownEnded>(packet.refSkillId()), std::chrono::milliseconds(skill.actionReuseDelay));
          skillsOnCooldown_.emplace(packet.refSkillId());
        } else {
          LOG(Info) << "  Executing step 2 or greater of a chain skill. Not going to set a cooldown\n";
        }
        const auto castDuration = skill.actionPreparingTime + skill.actionCastingTime;
        const auto totalDuration = gameData_.skillData().getSkillTotalDuration(packet.refSkillId());
        LOG(Info) << "  Cast time " << castDuration << ", total time " << totalDuration << '\n';

        if (activeChainSkillId_) {
          // Executing step 2 or greater of a chain skill
          if (activeAction_ &&
              activeAction_->commandType == packet::enums::CommandType::kExecute &&
              activeAction_->actionType == packet::enums::ActionType::kCast) {
            // const auto chainStartSkill = gameData_.skillData().getSkillById(activeAction_->refSkillId);
            LOG(Info) << "  Started chain with skill " << activeAction_->refSkillId << '\n';
            // TODO: Verify that we can get to this Id from the beginning skill
          } else {
            LOG(Error) << "  Executing step of chain, but the active action doesnt match\n";
          }
          const auto activeChainSkill = gameData_.skillData().getSkillById(*activeChainSkillId_);
          if (activeChainSkill.basicChainCode != packet.refSkillId()) {
            LOG(Error) << "  Active chain skill says " << activeChainSkill.basicChainCode << " is next, but we're on " << packet.refSkillId() << '\n';
          }
          LOG(Info) << "  Next step of chain " << *activeChainSkillId_ << "->" << packet.refSkillId() << '\n';
          activeChainSkillId_ = packet.refSkillId();
        } else {
          // Normal skill, or the first in a chain
          if (skill.isInstant()) {
            // This skill can be cast while another is being cast
            LOG(Info) << "  Instant skill, saving cast " << packet.castId() << ", " << packet.refSkillId() << '\n';
            saveCastAndPrint(packet.castId(), packet.refSkillId());
            // Remove from queued instant skill list
            queuedInstantSkills_.erase(packet.refSkillId());
          } else {
            if (queuedAction_) {
              if (queuedAction_->actionType == packet::enums::ActionType::kCast) {
                if (queuedAction_->refSkillId == packet.refSkillId()) {
                  if (skill.isPseudoinstant()) {
                    // TODO: I dont know if this is better here or in AC END
                    LOG(Info) << "  Pseudoinstant skill, saving cast " << packet.castId() << "," << packet.refSkillId() << " and discarding queued skill\n";
                    queuedAction_.reset();
                  } else {
                    if (activeAction_) {
                      LOG(Warning) << "  Overwriting active action\n";
                    }
                    LOG(Info) << "  Moving queued action to active\n";
                    activeAction_ = queuedAction_;
                    queuedAction_.reset();
                    if (skill.basicChainCode != 0) {
                      // Another skill will follow in the chain, set this part as the active chain skill
                      activeChainSkillId_ = packet.refSkillId();
                      LOG(Info) << "  First skill of the chain\n";
                    }
                  }
                } else {
                  LOG(Error) << "  Executing a skill that is not our queued action\n";
                }
              } else {
                LOG(Error) << "  Not a cast action queued?\n";
              }
            } else {
              LOG(Error) << "  No queued action\n";
              // // Build active action
              // activeAction_.emplace();
              // activeAction_->commandType = packet::enums::
            }
            if (skill.isTele()) {
              // Teleport skill. Lightning dash, wizard tele, or warrior sprint
              // Dont expect an end
              LOG(Info) << "  Teleport skill, no END coming\n";
            } else {
              // TODO: Some skills dont have an end. Which ones?!?
              //  Sometimes the beginning of the chain doesnt have an end
              LOG(Info) << "  End coming for this skill\n";
              saveCastAndPrint(packet.castId(), packet.refSkillId());
            }
          }
        }

        if (queuedCommonAttack_) {
          LOG(Info) << "  Have a queued common attack. Destroying it\n";
          queuedCommonAttack_.reset();
        }
      }
      printQueues();
    }
  }
}

// =====================================================================================================================
// =====================================================[%5d]   END=====================================================
// =====================================================================================================================
void SkillUseModule::serverAgentSkillEndReceived(packet::parsing::ServerAgentSkillEnd &packet) {
  if (packet.result() == 1) {
    LOG(Info);
    printf("[%5d]   END\n", packet.castId());
    for (const auto &hitObject : packet.action().hitObjects) {
      for (const auto &hit : hitObject.hits) {
        if (hit.hitResult == packet::enums::HitResult::kKill &&
            hitObject.objGlobalId == targetGId_) {
          std::cout << "Target died " << (int)hit.damageFlag << ' ' << hit.damage << "\n";
          // Our target died
          targetDied();
        } else if ((hitObject.objGlobalId == selfState_.globalId()) && hit.hitResult == packet::enums::HitResult::kKnockdown) {
          // knockedDown();
        } else if ((hitObject.objGlobalId == selfState_.globalId()) && hit.hitResult == packet::enums::HitResult::kKnockback) {
          knockedBack();
        }
      }
    }
    auto it = castSkillMap_.find(packet.castId());
    if (it == castSkillMap_.end()) {
      // Cast by someone else or we forgot to track it
    } else {
      if (activeChainSkillId_) {
        // Actively executing a chain (this was actually the end of the first part, as subsequent parts dont have an END)
        LOG(Info) << "  End of first skill in chain. Not doing anything\n";
      } else {
        const auto &skill = gameData_.skillData().getSkillById(it->second);
        if (skill.isInstant()) {
          LOG(Info) << "  Instant skill, not touching active action\n";
        } else if (skill.isPseudoinstant()) {
          LOG(Info) << "  Pseudoinstant skill, not touching active action\n";
        } else {
          // if (skill.actionAutoAttackType == 1 &&
          //     !queuedAction_ &&
          //     !queuedCommonAttack_) {
          //   LOG(Info) << "  Skill ended. No queued skill. To be followed by common attack. Queueing\n";
          //   // Copy active action to get same caster, target, etc.
          //   if (activeAction_) {
          //     queuedCommonAttack_ = activeAction_;
          //   } else {
          //     LOG(Error) << "  Skill ended but no active action!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
          //   }
          //   queuedCommonAttack_->actionType = packet::enums::ActionType::kAttack;
          // }
          LOG(Info) << "  Unsetting active action\n";
          activeAction_.reset();
        }
      }
      castSkillMap_.erase(it);
      LOG(Info) << "  Cast skill map [ ";
      for (const auto castSkill : castSkillMap_) {
        std::cout << '(' << castSkill.first << ',' << castSkill.second << "), ";
      }
      std::cout << "]\n";
  //     const auto skill = gameData_.skillData().getSkillById(skillId);
  //     if (skill.basicChainCode != 0) {
  //       // Another skill comes in the chain
  //       if (!activeServerSkill_) {
  //         std::cout << "!!!!!!!!!!!!!!!Doing a chain skill, but had no active skill!!!!!!!!!!!!!!!\n";
  //       }
  //       activeServerSkill_ = skill.basicChainCode;
  //     }
      printQueues();
    }
  }
}

void SkillUseModule::printBuffs() {
  std::cout << "Buffs:\n";
  for (const auto &buff : activeBuffs_) {
    printf(" %6d %6d %s\n",buff.skillRefId, buff.token, gameData_.skillData().getSkillById(buff.skillRefId).basicCode.c_str());
  }
}

void SkillUseModule::printQueues() {
  LOG(Info) << "  -->Queues<--\n";
  LOG(Info) << "      Pending: [";
  for (const auto &action : pendingActionQueue_) {
    print(action);
    std::cout << ',';
  }
  std::cout << "]\n";
  LOG(Info) << "      Queued common attack: [";
  if (queuedCommonAttack_) {
    print(*queuedCommonAttack_);
  }
  std::cout << "]\n";
  LOG(Info) << "      Instant: [";
  for (const auto &skillId : queuedInstantSkills_) {
    std::cout << skillId << ',';
  }
  std::cout << "]\n";
  LOG(Info) << "      Queued: [";
  if (queuedAction_) {
    print(*queuedAction_);
  }
  std::cout << "]\n";
  LOG(Info) << "      Active: [";
  if (activeAction_) {
    print(*activeAction_);
  }
  std::cout << "]\n";
  LOG(Info) << "      Active chain: [";
  if (activeChainSkillId_) {
    std::cout << *activeChainSkillId_;
  }
  std::cout << "]\n";
}

void SkillUseModule::requestedAction(const packet::structures::ActionCommand &actionCommand) {
  pendingActionQueue_.emplace_back(actionCommand);
}

void SkillUseModule::clientAgentActionCommandRequestReceived(packet::parsing::ClientAgentActionCommandRequest &packet) {
  LOG(Info) << ">>>>> Action command\n";
  // TODO: Should we block this if there's something already in the queue? Unless its an instant

  if (packet.commandType() == packet::enums::CommandType::kExecute) {
    if (packet.actionType() == packet::enums::ActionType::kCast) {
      if (packet.targetType() == packet::enums::TargetType::kEntity) {
        LOG(Info) << "  Cast " << packet.refSkillId() << " on " << packet.targetGlobalId() << "\n";
      } else {
        LOG(Info) << "  Cast " << packet.refSkillId() << "\n";
      }
      requestedAction(packet.actionCommand());
      printQueues();
    } else if (packet.actionType() == packet::enums::ActionType::kDispel) {
      LOG(Info) << "  Dispel " << packet.refSkillId() << '\n';
    } else if (packet.actionType() == packet::enums::ActionType::kTrace) {
      LOG(Info) << "  Trace\n";
    } else if (packet.actionType() == packet::enums::ActionType::kAttack) {
      LOG(Info) << "  Common attack\n";
      requestedAction(packet.actionCommand());
      printQueues();
    } else if (packet.actionType() == packet::enums::ActionType::kPickup) {
      LOG(Info) << "  Pickup\n";
    }
  } else if (packet.commandType() == packet::enums::CommandType::kCancel) {
    LOG(Info) << "  Cancel\n";
  }
}

void SkillUseModule::serverAgentBuffAddReceived(packet::parsing::ServerAgentBuffAdd &packet) {
  if (packet.globalId() == selfState_.globalId()) {
    printf("[%13lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    std::cout << "Buff " << packet.skillRefId() << " added to us. Active buff token: " << packet.activeBuffToken() << '\n';
    if (packet.activeBuffToken() != 0) {
      // TODO: Better understand token == 0
      activeBuffs_.emplace_back(packet.skillRefId(), packet.activeBuffToken());
    }
    printBuffs();
  // } else {
  //   std::cout << "Buff " << packet.skillRefId() << " added to " << packet.globalId() << ". Active buff token: " << packet.activeBuffToken() << '\n';
  }
  const auto skill = gameData_.skillData().getSkillById(packet.skillRefId());
}

void SkillUseModule::serverAgentBuffRemoveReceived(packet::parsing::ServerAgentBuffRemove &packet) {
  const auto &tokens = packet.tokens();
  printf("[%13lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
  std::cout << "Buffs with tokens [ ";
  for (auto i : packet.tokens()) {
    std::cout << i << ", ";
  }
  std::cout << "] removed\n";
  activeBuffs_.erase(std::remove_if(activeBuffs_.begin(), activeBuffs_.end(), [&tokens](const Buff &buff) { return std::find(tokens.begin(), tokens.end(), buff.token) != tokens.end(); }), activeBuffs_.end());
  printBuffs();
}

void SkillUseModule::serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) {
  // std::cout << "Entity state updated for " << packet.gId() << ", stateType = " << static_cast<uint32_t>(packet.stateType()) << ", state = " << static_cast<uint32_t>(packet.state()) << '\n';
  if (selfState_.spawned() && packet.gId() == selfState_.globalId()) {
    if (packet.stateType() == packet::parsing::StateType::kBodyState) {
      if (static_cast<packet::enums::BodyState>(packet.state()) == packet::enums::BodyState::kNormal) {
        // We're no longer untouchable!
        // TODO: Track that we were untouchable before
        //  Incorrectly get this message when coming out of stealth
        broker_.injectPacket(packet::building::ServerAgentChatUpdate::notice("No longer untouchable"), PacketContainer::Direction::kServerToClient);
      }
    } else if (packet.stateType() == packet::parsing::StateType::kLifeState) {
      const auto lifeState = static_cast<packet::enums::LifeState>(packet.state());
      if (lifeState == packet::enums::LifeState::kDead) {
        died();
      }
    }
  }
  if (packet.gId() == selfState_.globalId() && packet.stateType() == packet::parsing::StateType::kMotionState) {
    std::cout << "Updated MotionState " << static_cast<uint32_t>(packet.state()) << '\n';
  }
}

void SkillUseModule::tryCastNext() {
  // if (!tryCastSkills_.empty()) {
  //   auto skillId = tryCastSkills_.front();
  //   tryCastSkills_.pop_front();
  //   printf("[%13lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
  //   std::cout << "Trying to cast " << skillId << '\n';
  //   broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::cast(skillId), PacketContainer::Direction::kClientToServer);
  //   requestedAction(skillId);
  //   eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTemp), std::chrono::milliseconds(15));
  // }
}

void SkillUseModule::activeActionStarted() {
  if (activeAction_) {
    if (activeAction_->actionType == packet::enums::ActionType::kCast) {
      // Cant get here with an instant skill
      // const auto &skill = gameData_.skillData().getSkillById(activeAction_->refSkillId);
      // eventBroker_.publishDelayedEvent(std::make_unique<event::SkillCooldownEnded>(activeAction_->refSkillId), std::chrono::milliseconds(skill.actionReuseDelay));
      // skillsOnCooldown_.emplace(activeAction_->refSkillId);
      // const auto totalDuration = gameData_.skillData().getSkillTotalDuration(activeAction_->refSkillId);
      // constexpr int kMsBefore = 150;
      // eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kSkillCastAboutToEnd), std::chrono::milliseconds(std::max(totalDuration-kMsBefore, 0)));
      // waitingForCast_ = true;
    }
  }
}

/*
queue success
if (is skill or dispel) {
  put into queue
} else {
  // common attack, pick, trace
  put into active
}

skill begin
if (is skill in queue) {
  move into active
}

skill end
if (is skill in active) {
  remove from active
}
*/

void SkillUseModule::serverAgentActionCommandResponseReceived(packet::parsing::ServerAgentActionCommandResponse &packet) {
  constexpr int kMsBefore = 350;
  // TODO: Subscribe to all skills used so we can track all cooldowns
  if (packet.actionState() == packet::parsing::ActionState::kBegin) {
    // =====================================================================================================================
    // ======================================================AC QUEUE=======================================================
    // =====================================================================================================================
    LOG(Info) << "<<<<< AC QUEUE, repeat: " << packet.repeatAction() << ", queue size:" << pendingActionQueue_.size() << '\n';
    // Cast
    //  Has Begin
    //  Has End, if isnt an instant skill
    // Dispel
    //  Has Begin
    //  Has End
    // Trace
    //  Has Begin
    //  Ends when target is gone or cancelled
    // Common attack
    //  Has Begin
    //  Ends when canceled or target dies
    //  Ends when another skill is cast (but after that skill begins)
    // Pickup
    //  Has Begin
    //  End when picked, despawns, or cancelled
    // Cancel
    //  Has no Begin
    //  Only receive End if no error
    if (!pendingActionQueue_.empty()) {
      const auto &action = pendingActionQueue_.front();
      pendingActionQueue_.pop_front();
      if (action.actionType == packet::enums::ActionType::kCast) {
        // Only dealing with skill casting now
        const auto &skill = gameData_.skillData().getSkillById(action.refSkillId);
        if (skill.isInstant()) {
          // Instant skill successfully "queued". It should be cast shortly
          LOG(Info) << "  Instant skill\n";
          // Track the skill use so that we dont spam it while waiting for the cast
          queuedInstantSkills_.emplace(action.refSkillId);
        } else {
          // Non instant skill queued
          if (queuedAction_) {
            LOG(Info) << "  Overwriting queued action\n";
          } else {
            LOG(Info) << "  Queueing action\n";
          }
          queuedAction_ = action;
        }
      } else if (action.actionType == packet::enums::ActionType::kAttack) {
        LOG(Info) << "  Queueing common attack\n";
        if (queuedCommonAttack_) {
          LOG(Info) << "  Already have a queued common attack, overwriting\n";
        }
        queuedCommonAttack_ = action;
      } else {
        LOG(Warning) << "  Queueing an action that isnt a cast or common attack\n";
      }
    } else {
      LOG(Warning) << "  Queueing, but no pending cast or common attack\n";
    }
    printQueues();
  } else if (packet.actionState() == packet::parsing::ActionState::kEnd) {
    // =====================================================================================================================
    // ======================================================AC   END=======================================================
    // =====================================================================================================================
    // Done using skill
    LOG(Info) << "<<<<< AC   END, repeat: " << (packet.repeatAction() ? "true" : "false") << '\n';
    if (activeAction_ &&
        activeAction_->commandType == packet::enums::CommandType::kExecute &&
        activeAction_->actionType == packet::enums::ActionType::kAttack) {
      LOG(Info) << "  Was common attacking. Nevermore\n";
      activeAction_.reset();
    } else if (activeAction_ &&
               activeAction_->commandType == packet::enums::CommandType::kExecute &&
               activeAction_->actionType == packet::enums::ActionType::kCast &&
               gameData_.skillData().getSkillById(activeAction_->refSkillId).isTele()) {
      // Teleport skill ended
      LOG(Info) << "  Teleport skill ended. Clearing active action\n";
      activeAction_.reset();
    } else if (activeAction_ &&
               activeAction_->commandType == packet::enums::CommandType::kExecute &&
               activeAction_->actionType == packet::enums::ActionType::kCast &&
               activeChainSkillId_) {
      LOG(Info) << "  This ends a chain\n";
      const auto skill = gameData_.skillData().getSkillById(activeAction_->refSkillId);
      if (skill.actionAutoAttackType == 1 &&
          !queuedAction_ &&
          !queuedCommonAttack_) {
        LOG(Info) << "  Chain ended. No queued skill. To be followed by common attack. Queueing\n";
        // Copy active action to get same caster, target, etc.
        queuedCommonAttack_ = activeAction_;
        queuedCommonAttack_->actionType = packet::enums::ActionType::kAttack;
        const int32_t kPrickSkillId = 9943;
        LOG(Info) << "  Also, going to try to cast another skill, " << kPrickSkillId << "\n";
        packet::structures::ActionCommand action;
        broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::cast(kPrickSkillId, activeAction_->targetGlobalId), PacketContainer::Direction::kClientToServer);
        action = createActionCommandCastSkill(kPrickSkillId);
        requestedAction(action);
      }
      LOG(Info) << "  Unsetting active action and chain\n";
      activeAction_.reset();
      activeChainSkillId_.reset();
    } else if (activeAction_) {
      LOG(Warning) << "  Have an active action\n";
    }
    printQueues();
    // if (!pendingActionQueue_.empty() && pendingActionQueue_.front().commandType == packet::enums::CommandType::kCancel) {
    //   std::cout << "Pop the cancel action\n";
    //   pendingActionQueue_.pop_front();
    // } else {
    //   // Do we always have an active action?
    //   if (!activeAction_) {
    //     std::cout << "  No active action!\n";
    //   } else {
    //     if (queuedAction_) {
    //       // Assuming that the queued action immediately begins (TODO: Verify)
    //       std::cout << "  Moving queued action as active\n";
    //       activeAction_ = queuedAction_;
    //       if (activeAction_->actionType == packet::enums::ActionType::kCast) {
    //         const auto &skill = gameData_.skillData().getSkillById(activeAction_->refSkillId);
    //         eventBroker_.publishDelayedEvent(std::make_unique<event::SkillCooldownEnded>(activeAction_->refSkillId), std::chrono::milliseconds(skill.actionReuseDelay));
    //         skillsOnCooldown_.emplace(activeAction_->refSkillId);
    //         const auto totalDuration = gameData_.skillData().getSkillTotalDuration(activeAction_->refSkillId);
    //         eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kSkillCastAboutToEnd), std::chrono::milliseconds(std::max(totalDuration-kMsBefore, 0)));
    //         waitingForCast_ = true;
    //         std::cout << "    Skill cooldown is set as " << skill.actionReuseDelay << "ms and waiting " << std::max(totalDuration-kMsBefore, 0) << "ms before trying to cast " << skill.actionPreparingTime << ',' << skill.actionCastingTime << ',' << skill.actionActionDuration << "\n";
    //       }
    //       queuedAction_.reset();
    //     } else {
    //       if (activeAction_->actionType == packet::enums::ActionType::kCast &&
    //           gameData_.skillData().getSkillById(activeAction_->refSkillId).actionAutoAttackType == 1) {
    //         // Will auto attack afterwards, change active action to common attack
    //         // TODO: Check if current target is still alive
    //         std::cout << "  Changing cast to common attack\n";
    //         activeAction_->actionType = packet::enums::ActionType::kAttack;
    //         // std::cout << "  Trying to cancel auto attack\n";
    //         // broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::cancel(), PacketContainer::Direction::kClientToServer);
    //         // packet::structures::ActionCommand action;
    //         // action.commandType = packet::enums::CommandType::kCancel;
    //         // requestedAction(action);
    //       } else {
    //         std::cout << "  Active action ended, no queued action\n";
    //         activeAction_.reset();
    //       }
    //       // Nothing to do
    //     }
    //   }
    // }
    // printQueues();
  } else if (packet.actionState() == packet::parsing::ActionState::kError) {
    printf("[%13lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    printf("<<<<< AC ERROR, %6d\n", packet.errorCode());
    if (!pendingActionQueue_.empty()) {
      std::cout << "  Popping ";
      print(pendingActionQueue_.front());
      std::cout << '\n';
      pendingActionQueue_.pop_front();
    } else {
      std::cout << "  Cant pop\n";
    }
    if (packet.errorCode() != 16388) {
      std::cout << "WHOA WHOA WHOA WHOA WHOA WHOA, new error code!!!! " << packet.errorCode() << '\n';
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
  std::regex selectGidRegex(R"delim((commonattack|trace|pickup|select) ([0-9]+))delim");
  std::regex startAttackingRegex(R"delim(attack ([0-9]+))delim");
  std::regex stopAttackingRegex(R"delim(pause)delim");
  std::regex castTestRegex(R"delim(try)delim");
  std::smatch regexMatch;
  if (std::regex_match(packet.message(), regexMatch, selectGidRegex)) {
    const std::string operation = regexMatch[1].str();
    state::Entity::EntityId entityId = std::stoi(regexMatch[2].str());
    if (operation == "select") {
      selectEntity(entityId);
    } else if (operation == "commonattack") {
      commonAttackEntity(entityId);
    } else if (operation == "trace") {
      traceEntity(entityId);
    } else if (operation == "pickup") {
      pickupEntity(entityId);
    }
    return false;
  } else if (std::regex_match(packet.message(), regexMatch, startAttackingRegex)) {
    std::cout << "=================================\n";
    std::cout << "======Chat. Starting attack======\n";
    std::cout << "=================================\n";
    targetGId_ = stol(regexMatch[1].str());
    attackMode_ = true;
    attack();
    return false;
  } else if (std::regex_match(packet.message(), regexMatch, stopAttackingRegex)) {
    std::cout << "================================\n";
    std::cout << "======Chat. Pausing attack======\n";
    std::cout << "================================\n";
    attackMode_ = false;
    return false;
  } else if (std::regex_match(packet.message(), regexMatch, castTestRegex)) {
    tryCastSkills_.clear();
    tryCastSkills_.push_back(8244);
    tryCastSkills_.push_back(7985);
    tryCastSkills_.push_back(7910);
    tryCastNext();
    return false;
  } else {
    return true;
  }
}

void SkillUseModule::serverAgentActionSelectResponseReceived(packet::parsing::ServerAgentActionSelectResponse &packet) {}

} // namespace module

/*
// Queued common attack but never used it. Cancelled perfectly
[    345328681] [Info] [ 1029]   END
[    345328682] [Info]   Skill ended. No queued skill. To be followed by common attack
[    345328682] [Info]   -->Queues<--
[    345328683] [Info]       Pending: []
[    345328683] [Info]       Queued common attack: []
[    345328683] [Info]       Instant: []
[    345328684] [Info]       Queued: []
[    345328684] [Info]       Active: [{Cast 7844,104800}]
[    345328685] [Info]   Queueing said common attack
[    345328685] [Info]   Unsetting active action
[    345328685] [Info]   Cast skill map [ ]
[    345328686] [Info]   -->Queues<--
[    345328686] [Info]       Pending: []
[    345328686] [Info]       Queued common attack: [{Attack,104800}]
[    345328687] [Info]       Instant: []
[    345328687] [Info]       Queued: []
[    345328688] [Info]       Active: []
[    345328742] [Info]   Waiting for skill cast completed
[    345328950] [Info] <<<<< AC   END, repeat: true
[    345328995] [Info] >>>>> Action command
[    345328996] [Info]   Cancel
need to process cancel to know to remove it from the queue?
[    345329029] [Info] <<<<< AC   END, repeat: false
[    345329073] [Info] <<<<< AC   END, repeat: false
[    345330891]   Skill 7844 cooldown ended

[    380887848] [Info] [ 1756] BEGIN  10418
[    380887849] [Info]   Setting skill 10418 cooldown as 30000ms
[    380887850] [Info]   Cast time 166, total time 7999
[    380887853] [Info]   Waiting 7849ms before trying to cast 166,0,1515
[    380888240] [Info] [ 1756]   END
[    380888934] [Info] [ 1757] BEGIN  10429
694
1086
[    380888936] [Info]   Cast time 0, total time 6318
[    380889964] [Info] [ 1758] BEGIN  10440
1030 skillData: 992
[    380889965] [Info]   Cast time 0, total time 5326
[    380891085] [Info] [ 1759] BEGIN  10451
1121 skillData: 993
[    380891087] [Info]   Cast time 0, total time 4333
[    380892129] [Info] [ 1760] BEGIN  10462
1044 skillData: 1006
[    380892130] [Info]   Cast time 0, total time 3327
[    380893208] [Info] [ 1761] BEGIN  10473
1079 skillData: 983
[    380893210] [Info]   Cast time 0, total time 2344
[    380894273] [Info] [ 1762] BEGIN  10484
1065 skillData: 1013
[    380894274] [Info]   Cast time 0, total time 1331
[    380895596] [Info] <<<<< AC   END, repeat: false
1323 skillData: 1331

total
7748

*/

/*
==========================================
=======Skills that dont have an END=======
==========================================

To test:
Reverse, Grad Reverse, Reverse Oblation/Immolation

Cleric:
11643 - Innocent
11649 - Integrity
11728 - Healing Favor
11719 - Healing Division
11805 - Group Recovery ->
11811 - Holy Group Recovery
11776 - Recovery ->
11783 - Holy Recovery

Bard:
11285 - Booming Wave
11264 - Weird Chord
30197 - Booming Chord
11248 - Horror Chord
11390 - Rave Harmony
*/