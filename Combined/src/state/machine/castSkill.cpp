#include "castSkill.hpp"
#include "moveItemInInventory.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "type_id/categories.hpp"

namespace state::machine {

std::optional<uint8_t> getInventorySlotOfWeaponForSkill(const pk2::ref::Skill &skillData, const Bot &bot) {
  // TODO: Skill might not require a weapon
  const uint8_t kWeaponInventorySlot{6};
  std::vector<type_id::TypeCategory> possibleWeapons;
  for (auto i : skillData.reqi()) {
    if (i.typeId3 != 6) {
      LOG() << "reqi asks for non-weapon (typeId3: " << static_cast<int>(i.typeId3) << ")" << std::endl;
    }
    possibleWeapons.push_back(type_id::categories::kEquipment.subCategory(i.typeId3).subCategory(i.typeId4));
  }
  if (skillData.reqCastWeapon1 != 255) {
    possibleWeapons.push_back(type_id::categories::kWeapon.subCategory(skillData.reqCastWeapon1));
  }
  if (skillData.reqCastWeapon2 != 255) {
    possibleWeapons.push_back(type_id::categories::kWeapon.subCategory(skillData.reqCastWeapon2));
  }
  if (possibleWeapons.empty()) {
    // Does not require a weapon.
    return {};
  }

  // First, check if the currently equipped weapon is valid for this skill
  if (bot.selfState().inventory.hasItem(kWeaponInventorySlot)) {
    const auto *item = bot.selfState().inventory.getItem(kWeaponInventorySlot);
    if (!item) {
      throw std::runtime_error("Have an item, but got null");
    }
    if (!type_id::categories::kWeapon.contains(item->typeData())) {
      throw std::runtime_error("Equipped \"weapon\" isnt a weapon");
    }
    if (item->isOneOf(possibleWeapons)) {
      // Currently equipped weapon can cast this skill
      return kWeaponInventorySlot;
    }
  }
  // Currently equipped weapon (if any) cannot cast this skill, search through our inventory for a weapon which can cast this skill
  std::vector<uint8_t> possibleWeaponSlots = bot.selfState().inventory.findItemsOfCategory(possibleWeapons);
  LOG() << "Possible slots with weapon for skill: [ ";
  for (const auto slot : possibleWeaponSlots) {
    std::cout << static_cast<int>(slot) << ", ";
  }
  std::cout << "]" << std::endl;
  if (possibleWeaponSlots.empty()) {
    throw std::runtime_error("We have no weapon that can cast this skill");
  }

  // TODO: Pick best option
  // For now, pick the first option
  return possibleWeaponSlots.front();
}

CastSkillStateMachineBuilder::CastSkillStateMachineBuilder(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId) : bot_(bot), skillRefId_(skillRefId) {
}

CastSkillStateMachineBuilder& CastSkillStateMachineBuilder::withTarget(sro::scalar_types::EntityGlobalId globalId) {
  targetGlobalId_ = globalId;
  return *this;
}

CastSkillStateMachineBuilder& CastSkillStateMachineBuilder::withWeapon(uint8_t weaponSlot) {
  weaponSlot_ = weaponSlot;
  return *this;
}

CastSkillStateMachineBuilder& CastSkillStateMachineBuilder::withShield(uint8_t shieldSlot) {
  shieldSlot_ = shieldSlot;
  return *this;
}

CastSkillStateMachineBuilder& CastSkillStateMachineBuilder::withImbue(sro::scalar_types::ReferenceObjectId imbueSkillRefId) {
  imbueSkillRefId_ = imbueSkillRefId;
  return *this;
}

std::unique_ptr<StateMachine> CastSkillStateMachineBuilder::create() const {
  return std::make_unique<CastSkill>(bot_, skillRefId_, targetGlobalId_, weaponSlot_, shieldSlot_, imbueSkillRefId_);
}

// ========================================================================================================================================================================================================

CastSkill::CastSkill(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId, std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId, std::optional<uint8_t> weaponSlot, std::optional<uint8_t> shieldSlot, std::optional<sro::scalar_types::ReferenceObjectId> imbueSkillRefId) : StateMachine(bot), skillRefId_(skillRefId), targetGlobalId_(targetGlobalId), weaponSlot_(weaponSlot), shieldSlot_(shieldSlot), imbueSkillRefId_(imbueSkillRefId) {
  stateMachineCreated(kName);
  if (weaponSlot_ && *weaponSlot_ == kWeaponInventorySlot_) {
    // Weapon is already where it needs to be, dont move it
    weaponSlot_.reset();
  }
  if (shieldSlot_ && *shieldSlot_ == kShieldInventorySlot_) {
    // Shield is already where it needs to be, dont move it
    shieldSlot_.reset();
  }

  if (imbueSkillRefId_) {
    const auto &imbueSkill = bot_.gameData().skillData().getSkillById(*imbueSkillRefId_);
    if (imbueSkill.actionPreparingTime != 0) {
      throw std::runtime_error("Want to use imbue skill, but has a non-zero actionPreparingTime");
    }
    if (imbueSkill.actionCastingTime != 0) {
      throw std::runtime_error("Want to use imbue skill, but has a non-zero actionCastingTime");
    }
    if (imbueSkill.actionActionDuration != 0) {
      throw std::runtime_error("Want to use imbue skill, but has a non-zero actionActionDuration");
    }
    if (imbueSkill.targetRequired) {
      throw std::runtime_error("Want to use imbue skill, but requires a target");
    }
  }
}

CastSkill::~CastSkill() {
  if (skillCastTimeoutEventId_) {
    bot_.eventBroker().cancelDelayedEvent(*skillCastTimeoutEventId_);
    skillCastTimeoutEventId_.reset();
  }
  stateMachineDestroyed();
}

void CastSkill::onUpdate(const event::Event *event) {
  if (done()) {
    return;
  }

  bool megaDebug{false};
  if (event != nullptr && event->eventCode == event::EventCode::kStateMachineActiveTooLong) {
    const auto maybeSkillName = bot_.gameData().getSkillNameIfExists(skillRefId_);
    LOG() << "We seem to be stuck. Skill name: " << (maybeSkillName ? *maybeSkillName : std::string("UNKNOWN")) << std::endl;
    LOG() << "expectingSkillCommandFailure_: " << expectingSkillCommandFailure_ << std::endl;
    LOG() << "skillCastTimeoutEventId_: " << skillCastTimeoutEventId_.has_value() << std::endl;
    LOG() << "waitingForSkillToEnd_: " << waitingForSkillToEnd_ << std::endl;
    LOG() << "done_: " << done_ << std::endl;
    megaDebug = true;
  }

  if (childState_) {
    // Have a child state, it takes priority
    childState_->onUpdate(event);
    if (childState_->done()) {
      childState_.reset();
    } else {
      // Dont execute anything else in this function until the child state is done
      return;
    }
  }

  if (event) {
    if (skillCastTimeoutEventId_ || waitingForSkillToEnd_) {
      // If we're not waiting for our skill to cast or end, there is no event we care about
      if (const auto *skillBeganEvent = dynamic_cast<const event::SkillBegan*>(event)) {
        if (skillBeganEvent->casterGlobalId == bot_.selfState().globalId) {
          // We cast this skill
          const auto rootSkillId = bot_.gameData().skillData().getRootSkillRefId(skillBeganEvent->skillRefId);
          if (rootSkillId != skillRefId_) {
            LOG() << "This isnt the skill we thought we were casting!!!" << std::endl;
            // TODO: How should we actually handle this error?
            // Note: This happens if our skill was queued too slowly and a common attack slipped between the cracks
            //  Lets just skip this
            return;
          }
          //  Should this data be stored inside "selfState"? "CurrentlyCastingSkill"? What about "instant" skills?
          //    Probably not, I think it should be in the event
          if (skillCastTimeoutEventId_) {
            bot_.eventBroker().cancelDelayedEvent(*skillCastTimeoutEventId_);
            skillCastTimeoutEventId_.reset();
          }
          waitingForSkillToEnd_ = true;
        }
      } else if (const auto *skillEndedEvent = dynamic_cast<const event::SkillEnded*>(event)) {
        if (skillEndedEvent->casterGlobalId == bot_.selfState().globalId && waitingForSkillToEnd_) {
          // We cast this skill
          const auto rootSkillId = bot_.gameData().skillData().getRootSkillRefId(skillEndedEvent->skillRefId);
          if (rootSkillId != skillRefId_) {
            LOG() << "This isnt the skill we thought we were casting!!! Thought we were casting " << skillRefId_ << " but this skill is " << skillEndedEvent->skillRefId << ", with root skill id " << rootSkillId << std::endl;
            // TODO: How should we actually handle this error?
            //  Probably ignore?
            return;
          }
          // This "skill" ended, but maybe it is only one piece of a chain. Figure out if more are coming
          LOG() << "Skill ended: " << skillEndedEvent->skillRefId << std::endl;
          bool wasLastPieceOfSkill{false};
          const auto &thisSkillData = bot_.gameData().skillData().getSkillById(skillEndedEvent->skillRefId);
          const bool isFinalPieceOfChain = (thisSkillData.basicChainCode == 0);
          if (isFinalPieceOfChain) {
            // There couldn't possibly be another piece
            LOG() << "Final piece of chain" << std::endl;
            wasLastPieceOfSkill = true;
          } else {
            LOG() << "Not final piece of chain " << thisSkillData.basicChainCode << std::endl;
            // This isn't the last piece of the chain, but maybe this piece did enough damage to kill the target
            // In this case, no other skill begin/ends will come
            // TODO: This block makes an assumption that this is an attack on another entity
            if (targetGlobalId_) {
              if (*targetGlobalId_ == bot_.selfState().globalId) {
                LOG() << "This skill is cast by us, on us" << std::endl;
              }
              if (const auto *characterEntity = dynamic_cast<const entity::Character*>(bot_.entityTracker().getEntity(*targetGlobalId_))) {
                if (characterEntity->knowCurrentHp()) {
                  if (characterEntity->currentHp() == 0) {
                    // Entity is dead
                    LOG() << "Entity is dead" << std::endl;
                    wasLastPieceOfSkill = true;
                  } else {
                    LOG() << "Entity has more HP" << std::endl;
                  }
                } else {
                  LOG() << "Dont know entity's HP" << std::endl;
                }
              }
            }
          }
          if (wasLastPieceOfSkill) {
            LOG() << "Is last piece of skill" << std::endl;
            waitingForSkillToEnd_ = false; // Do we need to set this if we're "done"?
            done_ = true;
            return;
          } else {
            LOG() << "This isnt the last piece of the skill; waiting" << std::endl;
          }
          // Note: There is a chance that this skill killed the target, but we dont know that it's dead yet
          //  We will try to cast a skill on it, but it will fail
        }
      } else if (const auto *commandError = dynamic_cast<const event::CommandError*>(event)) {
        if (commandError->command.commandType == packet::enums::CommandType::kExecute) {
          if (commandError->command.actionType == packet::enums::ActionType::kCast) {
            if (commandError->command.refSkillId == skillRefId_) {
              LOG() << "Our command to cast this skill failed" << std::endl;
              if (expectingSkillCommandFailure_) {
                LOG() << "Expecting this; not acting on it" << std::endl;
                expectingSkillCommandFailure_ = false;
              } else {
                if (skillCastTimeoutEventId_) {
                  bot_.eventBroker().cancelDelayedEvent(*skillCastTimeoutEventId_);
                  skillCastTimeoutEventId_.reset();
                }
                if (waitingForSkillToEnd_) {
                  throw std::runtime_error("Waiting for skill to cast, command failed, but we were waiting for the skill to end (which means it must have started)");
                }
              }
            }
          }
        }
      } else if (const auto *skillFailedEvent = dynamic_cast<const event::OurSkillFailed*>(event)) {
        if (skillFailedEvent->skillRefId != skillRefId_) {
          LOG() << "Received skill failed event for a skill we're not trying to cast (" << skillFailedEvent->skillRefId << ')' << std::endl;
        } else {
          LOG() << "Our skill (" << skillFailedEvent->skillRefId << "," << bot_.gameData().textItemAndSkillData().getSkillName(bot_.gameData().skillData().getSkillById(skillFailedEvent->skillRefId).uiSkillName) << ") failed to cast" << std::endl;
          expectingSkillCommandFailure_ = true;
          if (!skillCastTimeoutEventId_) {
            throw std::runtime_error("Failed to cast skill, but didn't have a timeout running for it");
          }
          bot_.eventBroker().cancelDelayedEvent(*skillCastTimeoutEventId_);
          skillCastTimeoutEventId_.reset();
          if (waitingForSkillToEnd_) {
            LOG() << "Weird, skill started, but failed?" << std::endl;
            waitingForSkillToEnd_ = false;
          }
        }
      } else if (event->eventCode == event::EventCode::kKnockedBack || event->eventCode == event::EventCode::kKnockedDown) {
        // We've been completely interruped, yield
        LOG() << "We've been knocked back while trying to cast a skill. Aborting state" << std::endl;
        done_ = true;
        return;
        // TODO: I think instead it makes sense to return a status, which says something like "Interrupted", then the parent can choose to destroy us
        // We were waiting for a skill to cast or finish, this cancelled our skill
        if (skillCastTimeoutEventId_) {
          bot_.eventBroker().cancelDelayedEvent(*skillCastTimeoutEventId_);
          skillCastTimeoutEventId_.reset();
        }
        waitingForSkillToEnd_ = false;
      } else if (const auto *skillCastTimeoutEvent = dynamic_cast<const event::SkillCastTimeout*>(event)) {
        if (skillCastTimeoutEvent->skillId == skillRefId_) {
          if (!skillCastTimeoutEventId_) {
            throw std::runtime_error("The skill we were constructed to cast has timed out, but we weren't tracking a timeout event");
          }
          skillCastTimeoutEventId_.reset();
          // Relinqush control to our master.
          LOG() << "Done casting skill because it timed out" << std::endl;
          done_ = true;
          return;
        }
      }
    }
  }

  if (bot_.selfState().stunnedFromKnockback || bot_.selfState().stunnedFromKnockdown) {
    // Cannot cast a skill right now
    if (megaDebug) LOG() << "stunned from kb or kd" << std::endl;
    return;
  }

  if (weaponSlot_) {
    // Need to move weapon, create child state to do so
    setChildStateMachine<MoveItemInInventory>(*weaponSlot_, kWeaponInventorySlot_);
    // We assume that the child state will complete successfully, so we will reset the weaponSlot_ here
    weaponSlot_.reset();
    onUpdate(event);
    return;
  }

  if (shieldSlot_) {
    // Need to move shield, create child state to do so
    setChildStateMachine<MoveItemInInventory>(*shieldSlot_, kShieldInventorySlot_);
    // We assume that the child state will complete successfully, so we will reset the shieldSlot_ here
    shieldSlot_.reset();
    onUpdate(event);
    return;
  }

  // At this point, the required equipment is equipped
  if (skillCastTimeoutEventId_ || waitingForSkillToEnd_) {
    // Still waiting on skill
    if (megaDebug) LOG() << "still waiting on skill" << std::endl;
    return;
  }

  // Cast skill
  // TODO: Handle common attack
  PacketContainer castSkillPacket;
  if (targetGlobalId_) {
    // Have a target
    castSkillPacket = packet::building::ClientAgentActionCommandRequest::cast(skillRefId_, *targetGlobalId_);
  } else {
    // No target
    castSkillPacket = packet::building::ClientAgentActionCommandRequest::cast(skillRefId_);
  }

  // Cast imbue if it exists and is not active
  if (imbueSkillRefId_) {
    LOG() << "Given imbue skill to cast" << std::endl;
    if (!targetGlobalId_) {
      throw std::runtime_error("Using an imbue when casting a skill on no target");
    }

    if (!bot_.selfState().buffIsActive(*imbueSkillRefId_)) {
      LOG() << "Imbue is not active." << std::endl;
      if (!bot_.selfState().skillEngine.skillIsOnCooldown(*imbueSkillRefId_)) {
        LOG() << "Imbue is not on cooldown; creating child CastSkill to cast it" << std::endl;
        CastSkillStateMachineBuilder builder(bot_, *imbueSkillRefId_);
        setChildStateMachine(builder.create());
        onUpdate(event);
        return;
      } else {
        const auto skillRemainingCooldown = bot_.selfState().skillEngine.skillRemainingCooldown(*imbueSkillRefId_, bot_.eventBroker());
        if (skillRemainingCooldown) {
          LOG() << "Imbue is on cooldown with " << skillRemainingCooldown->count() << "ms remaining" << std::endl;
        } else {
          LOG() << "We thought the imbue is on cooldown, but we don't know how much time is remaining...?" << std::endl;
        }
      }
      LOG() << "Imbue buff is not active. Waiting." << std::endl;
      return;
    } else {
      // Check the remaining duration of this buff, it it's less than the ping time, we ought to re-cast since it might expire before we cast the actual attack.
      LOG() << "Imbue is already active" << std::endl;
      constexpr const int kEstimatedPingMs = 100; // TODO: Get from a centralized location.
      const auto buffRemainingTime = bot_.selfState().buffMsRemaining(*imbueSkillRefId_);
      if (buffRemainingTime < kEstimatedPingMs) {
        LOG() << "Imbue might end before our skill-use packet even gets to the server (" << buffRemainingTime << " ms remaining)" << std::endl;
        const auto skillRemainingCooldown = bot_.selfState().skillEngine.skillRemainingCooldown(*imbueSkillRefId_, bot_.eventBroker());
        if (skillRemainingCooldown) {
          LOG() << "Skill remaining cooldown: " << skillRemainingCooldown->count() << "ms" << std::endl;
          if (skillRemainingCooldown->count() > kEstimatedPingMs) {
            LOG() << std::string(1000, '_') << "Too much cooldown. Waiting." << std::endl;
            return;
          }
        } else {
          LOG() << "Skill has no remaining cooldown" << std::endl;
        }
        // TODO: Add a data structure that tracks skills which were successfully cast, which should give us a buff, but for when we havent yet received BuffBegin. Like a "anticipated buffs" list.
        if (bot_.selfState().skillEngine.alreadyTriedToCastSkill(*imbueSkillRefId_)) {
          LOG() << "Already tried to cast skill.." << std::endl;
        }
        CastSkillStateMachineBuilder builder(bot_, *imbueSkillRefId_);
        setChildStateMachine(builder.create());
        onUpdate(event);
        // If the timing overlaps, the buff will not be removed and re-added. Instead, the new buff will be added, there will be a brief moment where we have two instances of the same buff, and then the old one will be removed.
        return;
      }
    }
  }

  LOG() << "Using skill " << bot_.gameData().textItemAndSkillData().getSkillName(bot_.gameData().skillData().getSkillById(skillRefId_).uiSkillName) << std::endl;
  if (bot_.selfState().skillEngine.skillIsOnCooldown(skillRefId_)) {
    LOG() << "  but skill is on cooldown" << std::endl;
  }
  bot_.packetBroker().injectPacket(castSkillPacket, PacketContainer::Direction::kClientToServer);
  skillCastTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent<event::SkillCastTimeout>(std::chrono::milliseconds(kSkillCastTimeoutMs), skillRefId_);
}

bool CastSkill::done() const {
  return done_;
}

} // namespace state::machine