#include "castSkill.hpp"
#include "moveItemInInventory.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "type_id/categories.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

namespace state::machine {

std::optional<uint8_t> getInventorySlotOfWeaponForSkill(const pk2::ref::Skill &skillData, const Bot &bot) {
  // TODO: Skill might not require a weapon
  const uint8_t kWeaponInventorySlot{6};
  std::vector<type_id::TypeCategory> possibleWeapons;
  for (auto i : skillData.reqi()) {
    if (i.typeId3 != 6) {
      LOG(INFO) << "reqi asks for non-weapon (typeId3: " << static_cast<int>(i.typeId3) << ")";
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
    if (!type_id::categories::kWeapon.contains(item->typeId())) {
      throw std::runtime_error("Equipped \"weapon\" isnt a weapon");
    }
    if (item->isOneOf(possibleWeapons)) {
      // Currently equipped weapon can cast this skill
      return kWeaponInventorySlot;
    }
  }
  // Currently equipped weapon (if any) cannot cast this skill, search through our inventory for a weapon which can cast this skill
  std::vector<uint8_t> possibleWeaponSlots = bot.selfState().inventory.findItemsOfCategory(possibleWeapons);
  LOG(INFO) << absl::StreamFormat("Possible slots with weapon for skill: [ %s ]", absl::StrJoin(possibleWeaponSlots, ", ", [](std::string *out, auto slot){
    out->append(std::to_string(slot));
  }));
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
    LOG(INFO) << "We seem to be stuck. Skill name: " << (maybeSkillName ? *maybeSkillName : std::string("UNKNOWN"));
    LOG(INFO) << "expectingSkillCommandFailure_: " << expectingSkillCommandFailure_;
    LOG(INFO) << "skillCastTimeoutEventId_: " << skillCastTimeoutEventId_.has_value();
    LOG(INFO) << "waitingForSkillToEnd_: " << waitingForSkillToEnd_;
    LOG(INFO) << "done_: " << done_;
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
            LOG(INFO) << "This isnt the skill we thought we were casting!!!";
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
            LOG(INFO) << "This isnt the skill we thought we were casting!!! Thought we were casting " << skillRefId_ << " but this skill is " << skillEndedEvent->skillRefId << ", with root skill id " << rootSkillId;
            // TODO: How should we actually handle this error?
            //  Probably ignore?
            return;
          }
          // This "skill" ended, but maybe it is only one piece of a chain. Figure out if more are coming
          LOG(INFO) << "Skill ended: " << skillEndedEvent->skillRefId;
          bool wasLastPieceOfSkill{false};
          const auto &thisSkillData = bot_.gameData().skillData().getSkillById(skillEndedEvent->skillRefId);
          const bool isFinalPieceOfChain = (thisSkillData.basicChainCode == 0);
          if (isFinalPieceOfChain) {
            // There couldn't possibly be another piece
            LOG(INFO) << "Final piece of chain";
            wasLastPieceOfSkill = true;
          } else {
            LOG(INFO) << "Not final piece of chain " << thisSkillData.basicChainCode;
            // This isn't the last piece of the chain, but maybe this piece did enough damage to kill the target
            // In this case, no other skill begin/ends will come
            // TODO: This block makes an assumption that this is an attack on another entity
            if (targetGlobalId_) {
              if (*targetGlobalId_ == bot_.selfState().globalId) {
                LOG(INFO) << "This skill is cast by us, on us";
              }
              if (const auto *characterEntity = dynamic_cast<const entity::Character*>(bot_.entityTracker().getEntity(*targetGlobalId_))) {
                if (characterEntity->knowCurrentHp()) {
                  if (characterEntity->currentHp() == 0) {
                    // Entity is dead
                    LOG(INFO) << "Entity is dead";
                    wasLastPieceOfSkill = true;
                  } else {
                    LOG(INFO) << "Entity has more HP";
                  }
                } else {
                  LOG(INFO) << "Dont know entity's HP";
                }
              }
            }
          }
          if (wasLastPieceOfSkill) {
            LOG(INFO) << "Is last piece of skill";
            waitingForSkillToEnd_ = false; // Do we need to set this if we're "done"?
            done_ = true;
            return;
          } else {
            LOG(INFO) << "This isnt the last piece of the skill; waiting";
          }
          // Note: There is a chance that this skill killed the target, but we dont know that it's dead yet
          //  We will try to cast a skill on it, but it will fail
        }
      } else if (const auto *commandError = dynamic_cast<const event::CommandError*>(event)) {
        if (commandError->command.commandType == packet::enums::CommandType::kExecute) {
          if (commandError->command.actionType == packet::enums::ActionType::kCast) {
            if (commandError->command.refSkillId == skillRefId_) {
              // Our command to cast this skill failed.
              if (expectingSkillCommandFailure_) {
                // Expecting this failure; not acting on it.
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
          LOG(INFO) << "Received skill failed event for a skill we're not trying to cast (" << skillFailedEvent->skillRefId << ')';
        } else {
          LOG(INFO) << "Our skill (" << skillFailedEvent->skillRefId << "," << bot_.gameData().textItemAndSkillData().getSkillName(bot_.gameData().skillData().getSkillById(skillFailedEvent->skillRefId).uiSkillName) << ") failed to cast";
          expectingSkillCommandFailure_ = true;
          if (!skillCastTimeoutEventId_) {
            throw std::runtime_error("Failed to cast skill, but didn't have a timeout running for it");
          }
          bot_.eventBroker().cancelDelayedEvent(*skillCastTimeoutEventId_);
          skillCastTimeoutEventId_.reset();
          if (waitingForSkillToEnd_) {
            LOG(INFO) << "Weird, skill started, but failed?";
            waitingForSkillToEnd_ = false;
          }
        }
      } else if (event->eventCode == event::EventCode::kKnockedBack || event->eventCode == event::EventCode::kKnockedDown) {
        // We've been completely interruped, yield
        LOG(INFO) << "We've been knocked back while trying to cast a skill. Aborting state";
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
          LOG(INFO) << "Done casting skill because it timed out";
          done_ = true;
          return;
        }
      }
    }
  }

  if (bot_.selfState().stunnedFromKnockback || bot_.selfState().stunnedFromKnockdown) {
    // Cannot cast a skill right now
    if (megaDebug) LOG(INFO) << "stunned from kb or kd";
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
    if (megaDebug) LOG(INFO) << "still waiting on skill";
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
    LOG(INFO) << "Given imbue skill to cast";
    if (!targetGlobalId_) {
      throw std::runtime_error("Using an imbue when casting a skill on no target");
    }

    if (!bot_.selfState().buffIsActive(*imbueSkillRefId_)) {
      LOG(INFO) << "Imbue is not active.";
      if (!bot_.selfState().skillEngine.skillIsOnCooldown(*imbueSkillRefId_)) {
        LOG(INFO) << "Imbue is not on cooldown; creating child CastSkill to cast it";
        CastSkillStateMachineBuilder builder(bot_, *imbueSkillRefId_);
        setChildStateMachine(builder.create());
        onUpdate(event);
        return;
      } else {
        const auto skillRemainingCooldown = bot_.selfState().skillEngine.skillRemainingCooldown(*imbueSkillRefId_, bot_.eventBroker());
        if (skillRemainingCooldown) {
          LOG(INFO) << "Imbue is on cooldown with " << skillRemainingCooldown->count() << "ms remaining";
        } else {
          LOG(INFO) << "We thought the imbue is on cooldown, but we don't know how much time is remaining...?";
        }
      }
      LOG(INFO) << "Imbue buff is not active. Waiting.";
      return;
    } else {
      // Check the remaining duration of this buff, it it's less than the ping time, we ought to re-cast since it might expire before we cast the actual attack.
      LOG(INFO) << "Imbue is already active";
      constexpr const int kEstimatedPingMs = 100; // TODO: Get from a centralized location.
      const auto buffRemainingTime = bot_.selfState().buffMsRemaining(*imbueSkillRefId_);
      if (buffRemainingTime < kEstimatedPingMs) {
        LOG(INFO) << "Imbue might end before our skill-use packet even gets to the server (" << buffRemainingTime << " ms remaining)";
        const auto skillRemainingCooldown = bot_.selfState().skillEngine.skillRemainingCooldown(*imbueSkillRefId_, bot_.eventBroker());
        if (skillRemainingCooldown) {
          LOG(INFO) << "Skill remaining cooldown: " << skillRemainingCooldown->count() << "ms";
          if (skillRemainingCooldown->count() > kEstimatedPingMs) {
            LOG(INFO) << std::string(1000, '_') << "Too much cooldown. Waiting.";
            return;
          }
        } else {
          LOG(INFO) << "Skill has no remaining cooldown";
        }
        // TODO: Add a data structure that tracks skills which were successfully cast, which should give us a buff, but for when we havent yet received BuffBegin. Like a "anticipated buffs" list.
        if (bot_.selfState().skillEngine.alreadyTriedToCastSkill(*imbueSkillRefId_)) {
          LOG(INFO) << "Already tried to cast skill..";
        }
        CastSkillStateMachineBuilder builder(bot_, *imbueSkillRefId_);
        setChildStateMachine(builder.create());
        onUpdate(event);
        // If the timing overlaps, the buff will not be removed and re-added. Instead, the new buff will be added, there will be a brief moment where we have two instances of the same buff, and then the old one will be removed.
        return;
      }
    }
  }

  LOG(INFO) << "Using skill " << bot_.gameData().textItemAndSkillData().getSkillName(bot_.gameData().skillData().getSkillById(skillRefId_).uiSkillName);
  if (bot_.selfState().skillEngine.skillIsOnCooldown(skillRefId_)) {
    LOG(INFO) << "  but skill is on cooldown";
  }
  bot_.packetBroker().injectPacket(castSkillPacket, PacketContainer::Direction::kClientToServer);
  skillCastTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent<event::SkillCastTimeout>(std::chrono::milliseconds(kSkillCastTimeoutMs), skillRefId_);
}

bool CastSkill::done() const {
  return done_;
}

} // namespace state::machine