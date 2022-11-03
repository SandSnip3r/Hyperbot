#include "castSkill.hpp"
#include "moveItemInInventory.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"

namespace state::machine {

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

std::unique_ptr<StateMachine> CastSkillStateMachineBuilder::create() const {
  return std::make_unique<CastSkill>(bot_, skillRefId_, targetGlobalId_, weaponSlot_, shieldSlot_);
}

// ========================================================================================================================================================================================================

CastSkill::CastSkill(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId, std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId, std::optional<uint8_t> weaponSlot, std::optional<uint8_t> shieldSlot) : StateMachine(bot), skillRefId_(skillRefId), targetGlobalId_(targetGlobalId), weaponSlot_(weaponSlot), shieldSlot_(shieldSlot) {
  if (weaponSlot_ && *weaponSlot_ == kWeaponInventorySlot_) {
    // Weapon is already where it needs to be, dont move it
    weaponSlot_.reset();
  }
  if (shieldSlot_ && *shieldSlot_ == kShieldInventorySlot_) {
    // Shield is already where it needs to be, dont move it
    shieldSlot_.reset();
  }
}

void CastSkill::onUpdate(const event::Event *event) {
  LOG() << "Castskill on update" << std::endl;
TODO_REMOVE_THIS_LABEL:
  if (childState_) {
    LOG() << "Descending into child on update" << std::endl;
    // Have a child state, it takes priority
    childState_->onUpdate(event);
    if (childState_->done()) {
      LOG() << "Child state is done" << std::endl;
      childState_.reset();
    } else {
      // Dont execute anything else in this function until the child state is done
      return;
    }
  }
  LOG() << "No child state" << std::endl;

  if (event) {
    if (waitingForSkillToCast_ || waitingForSkillToEnd_) {
      // If we're not waiting for our skill to cast or end, there is no event we care about
      if (const auto *skillBeganEvent = dynamic_cast<const event::SkillBegan*>(event)) {
        if (skillBeganEvent->casterGlobalId == bot_.selfState().globalId) {
          // We cast this skill
          // TODO: Check that this is the skill that we tried to cast
          //  Should this data be stored inside "selfState"? "CurrentlyCastingSkill"? What about "instant" skills?
          //    Probably not, I think it should be in the event
          waitingForSkillToCast_ = false;
          waitingForSkillToEnd_ = true;
        }
      } else if (const auto *skillEndedEvent = dynamic_cast<const event::SkillEnded*>(event)) {
        if (skillEndedEvent->casterGlobalId == bot_.selfState().globalId) {
          // We cast this skill
          // TODO: Check that this is the skill that we cast
          waitingForSkillToEnd_ = false; // Do we need to set this if we're "done"?
          done_ = true;
          return;
          // Note: There is a chance that this skill killed the target, but we dont know that it's dead yet
          //  We will try to cast a skill on it, but it will fail
        }
      } else if (const auto *commandError = dynamic_cast<const event::CommandError*>(event)) {
        if (commandError->command.commandType == packet::enums::CommandType::kExecute) {
          if (commandError->command.actionType == packet::enums::ActionType::kCast) {
            if (commandError->command.refSkillId == skillRefId_) {
              LOG() << "Our command to cast this skill failed" << std::endl;
              waitingForSkillToCast_ = false;
              if (waitingForSkillToEnd_) {
                throw std::runtime_error("Waiting for skill to cast, command failed, but we were waiting for the skill to end (which means it must have started)");
              }
              ++failCount_;
              if (failCount_ == kMaxFails_) {
                LOG() << "Reached max failure count, not going to try again!" << std::endl;
              }
            }
          }
        }
      } else if (event->eventCode == event::EventCode::kOurSkillFailed) {
        LOG() << "Our skill failed to cast" << std::endl;
        waitingForSkillToCast_ = false;
        if (waitingForSkillToEnd_) {
          LOG() << "Weird, skill started, but failed?" << std::endl;
          waitingForSkillToEnd_ = false;
        }
        ++failCount_;
        if (failCount_ == kMaxFails_) {
          LOG() << "Reached max failure count, not going to try again!" << std::endl;
        }
      }
    }
  }

  if (weaponSlot_) {
    // Need to move weapon, create child state to do so
    LOG() << "make_unique<MoveItemInInventory" << std::endl;
    childState_ = std::make_unique<MoveItemInInventory>(bot_, *weaponSlot_, kWeaponInventorySlot_);
    LOG() << "Created" << std::endl;
    // We assume that the child state will complete successfully, so we will reset the weaponSlot_ here
    weaponSlot_.reset();
    goto TODO_REMOVE_THIS_LABEL;
  }

  if (shieldSlot_) {
    // Need to move shield, create child state to do so
    LOG() << "make_unique<MoveItemInInventory" << std::endl;
    childState_ = std::make_unique<MoveItemInInventory>(bot_, *shieldSlot_, kShieldInventorySlot_);
    LOG() << "Created" << std::endl;
    // We assume that the child state will complete successfully, so we will reset the shieldSlot_ here
    shieldSlot_.reset();
    goto TODO_REMOVE_THIS_LABEL;
  }

  // At this point, the required equipment is equipped
  if (waitingForSkillToCast_ || waitingForSkillToEnd_) {
    return;
  }

  if (failCount_ >= kMaxFails_) {
    // Failed too many times, not going to try again
    // return; // For now, infinite retry is ok
    // TODO: Fix. The fact that it fails is a problem, ideally, we don't want to get here
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
  LOG() << "Using skill " << bot_.gameData().textItemAndSkillData().getSkillName(bot_.gameData().skillData().getSkillById(skillRefId_).uiSkillName) << std::endl;
  bot_.packetBroker().injectPacket(castSkillPacket, PacketContainer::Direction::kClientToServer);
  waitingForSkillToCast_ = true;
}

bool CastSkill::done() const {
  return done_;
}

} // namespace state::machine