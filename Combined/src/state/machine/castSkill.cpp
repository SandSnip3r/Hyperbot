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
    // Does not require a weapon
    LOG() << "Skill does not require a weapon" << std::endl;
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

std::unique_ptr<StateMachine> CastSkillStateMachineBuilder::create() const {
  return std::make_unique<CastSkill>(bot_, skillRefId_, targetGlobalId_, weaponSlot_, shieldSlot_);
}

// ========================================================================================================================================================================================================

CastSkill::CastSkill(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId, std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId, std::optional<uint8_t> weaponSlot, std::optional<uint8_t> shieldSlot) : StateMachine(bot), skillRefId_(skillRefId), targetGlobalId_(targetGlobalId), weaponSlot_(weaponSlot), shieldSlot_(shieldSlot) {
  stateMachineCreated(kName);
  if (weaponSlot_ && *weaponSlot_ == kWeaponInventorySlot_) {
    // Weapon is already where it needs to be, dont move it
    weaponSlot_.reset();
  }
  if (shieldSlot_ && *shieldSlot_ == kShieldInventorySlot_) {
    // Shield is already where it needs to be, dont move it
    shieldSlot_.reset();
  }
}

CastSkill::~CastSkill() {
  stateMachineDestroyed();
}

void CastSkill::onUpdate(const event::Event *event) {
  bool megaDebug{false};
  if (event != nullptr && event->eventCode == event::EventCode::kStateMachineActiveTooLong) {
    LOG() << "We seem to be stuck." << std::endl;
    LOG() << "expectingSkillCommandFailure_: " << expectingSkillCommandFailure_ << std::endl;
    LOG() << "waitingForSkillToCast_: " << waitingForSkillToCast_ << std::endl;
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
    if (waitingForSkillToCast_ || waitingForSkillToEnd_) {
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
          waitingForSkillToCast_ = false;
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
          LOG() << "End! " << skillEndedEvent->skillRefId << std::endl;
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
                waitingForSkillToCast_ = false;
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
          waitingForSkillToCast_ = false;
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
        waitingForSkillToCast_ = false;
        waitingForSkillToEnd_ = false;
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
  if (waitingForSkillToCast_ || waitingForSkillToEnd_) {
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
  LOG() << "Using skill " << bot_.gameData().textItemAndSkillData().getSkillName(bot_.gameData().skillData().getSkillById(skillRefId_).uiSkillName) << std::endl;
  bot_.packetBroker().injectPacket(castSkillPacket, PacketContainer::Direction::kClientToServer);
  waitingForSkillToCast_ = true;
}

bool CastSkill::done() const {
  return done_;
}

} // namespace state::machine