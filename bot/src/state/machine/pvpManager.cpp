#include "pvpManager.hpp"

#include "bot.hpp"
#include "common/sessionId.hpp"
#include "packet/building/serverAgentEntityUpdateHwanLevel.hpp"
#include "state/machine/disableGmInvisible.hpp"
#include "state/machine/dispelActiveBuffs.hpp"
#include "state/machine/enablePvpMode.hpp"
#include "state/machine/ensureFullVitalsAndNoStatuses.hpp"
#include "state/machine/gmCommandSpawnAndPickItems.hpp"
#include "state/machine/gmWarpToPosition.hpp"
#include "state/machine/intelligenceActor.hpp"
#include "state/machine/login.hpp"
#include "state/machine/resurrectInPlace.hpp"
#include "state/machine/spawnAndUseRepairHammerIfNecessary.hpp"
#include "state/machine/useItem.hpp"
#include "state/machine/waitForAllCooldownsToEnd.hpp"
#include "state/machine/walking.hpp"
#include "type_id/categories.hpp"

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>

namespace state::machine {

namespace {

enum class Player { kPlayer1, kPlayer2 };

Player ourPlayer(const common::PvpDescriptor &pvpDescriptor, const sro::scalar_types::EntityGlobalId globalId) {
  if (pvpDescriptor.player1GlobalId == globalId) {
    return Player::kPlayer1;
  }
  if (pvpDescriptor.player2GlobalId == globalId) {
    return Player::kPlayer2;
  }
  throw std::runtime_error("We are not one of the players in the PVP");
}

} // namespace

PvpManager::PvpManager(Bot &bot, const CharacterLoginInfo &characterLoginInfo) : StateMachine(bot), characterLoginInfo_(characterLoginInfo) {
  LOG(INFO) << "PvpManager Constructed";
}

PvpManager::~PvpManager() {
}

Status PvpManager::onUpdate(const event::Event *event) {
  ZoneScopedN("PvpManager::onUpdate");
  if (!initialized_) {
    initialized_ = true;
    if (bot_.loggedIn()) {
      // This state machine expected to be created when we're not logged in.
      throw std::runtime_error("We should not be logged in when we start the PvpManager");
    }

    LOG(INFO) << characterLoginInfo_.characterName << ". Need to log in";
    setChildStateMachine<state::machine::Login>(characterLoginInfo_);
    return onUpdate(event);
  }

  // If the child state is Login, immediately delegate to it.
  if (childState_ != nullptr && dynamic_cast<Login*>(childState_.get()) != nullptr) {
    const Status status = childState_->onUpdate(event);
    if (status == Status::kNotDone) {
      // We're still logging in. Nothing else to do.
      return status;
    }
    // We just finished logging in.
    CHAR_LOG(INFO) << "Finished logging in";
    childState_.reset();
    resetAndNotifyReadyForAssignment();
    return Status::kNotDone;
  }

  // We are logged in at this point.

  // Before maybe delegating to our child state machine:
  // - Check to see if someone died.
  // - Check to see if our opponent is ready for pvp.
  // - Check to see if we received a resurrect option.
  if (event != nullptr) {
    if (const auto *entityDespawned = dynamic_cast<const event::EntityDespawned*>(event); entityDespawned != nullptr) {
      if (pvpDescriptor_ && entityDespawned->globalId == pvpDescriptor_->player1GlobalId || entityDespawned->globalId == pvpDescriptor_->player2GlobalId) {
        // TODO: Handle.
        throw std::runtime_error("Someone in our pvp despawned!");
      }
    } else if (const auto *lifeStateChanged = dynamic_cast<const event::EntityLifeStateChanged*>(event); lifeStateChanged != nullptr) {
      // Maybe someone died.
      // We don't care about this event if we're already resurrecting ourself.
      const bool childStateIsResurrect = childState_ != nullptr && dynamic_cast<ResurrectInPlace*>(childState_.get()) != nullptr;
      if (!childStateIsResurrect) {
        const std::shared_ptr<entity::PlayerCharacter> playerCharacter = bot_.entityTracker().getEntity<entity::PlayerCharacter>(lifeStateChanged->globalId);
        if (playerCharacter->lifeState == sro::entity::LifeState::kDead) {
          // Someone died.
          // If we were pvping and it was us or the opponent, the pvp is over.
          const bool pvping = [&]() -> bool {
            if (childState_ != nullptr) {
              if (dynamic_cast<IntelligenceActor*>(childState_.get()) != nullptr) {
                return true;
              }
            }
            return false;
          }();
          if (pvping) {
            if (!pvpDescriptor_) {
              throw std::runtime_error("We are pvping but don't have a pvp descriptor");
            }
            if (lifeStateChanged->globalId == pvpDescriptor_->player1GlobalId ||
                lifeStateChanged->globalId == pvpDescriptor_->player2GlobalId) {
              CHAR_LOG(INFO) << "Either we or our opponent died! The pvp is over. " << lifeStateChanged->globalId << " died, we are " << bot_.selfState()->globalId;
              // Call the child state once more, so that it may report the final state to the replay buffer.
              Status childStatus = childState_->onUpdate(event);
              childState_.reset();
              pvpDescriptor_.reset();
              // If it was the opponent who died, we can immediately report that we're ready for our next assignment.
              if (lifeStateChanged->globalId == getOpponentGlobalId()) {
                resetAndNotifyReadyForAssignment();
                return Status::kNotDone;
              }
              if (childStatus != Status::kDone) {
                // Sent a terminal state to the child state machine, it should have returned "Done".
                throw std::runtime_error("Child state machine did not return Done after receiving a terminal state");
              }
            }
          }
          if (lifeStateChanged->globalId == bot_.selfState()->globalId) {
            // If we died, we need to resurrect ourself, regardless of what we're doing.
            setChildStateMachine<ResurrectInPlace>(receivedResurrectionOption_);
            return onUpdate(event);
          }
        }
      }
    } else if (const auto *readyForPvpEvent = dynamic_cast<const event::ReadyForPvp*>(event); readyForPvpEvent != nullptr) {
      CHAR_VLOG(1) << "ReadyForPvp";
      const sro::scalar_types::EntityGlobalId opponentsGlobalId = getOpponentGlobalId();
      if (readyForPvpEvent->globalId == opponentsGlobalId) {
        CHAR_VLOG(1) << "Received ReadyForPvp for our opponent";
        opponentIsReady_ = true;
        if (weAreReady_) {
          return startPvp(event);
        } else {
          CHAR_VLOG(1) << "Opponent is ready for PVP, but we are not. Waiting for us to be ready.";
          return Status::kNotDone;
        }
      }
    } else if (const auto *resurrectOption = dynamic_cast<const event::ResurrectOption*>(event)) {
      if (childState_ != nullptr && dynamic_cast<ResurrectInPlace*>(childState_.get()) != nullptr) {
        // Not yet resurrecting in place. Take note that we received this so that when the ResurrectInPlace state machine is run, it knows not to wait for this.
        // We need to do this because the (resurrect options) and (died) packets can come in any order.
        if (resurrectOption->option != packet::enums::ResurrectionOptionFlag::kAtPresentPoint) {
          throw std::runtime_error("We can only handle resurrecting in place");
        }
        receivedResurrectionOption_ = true;
      }
    }
  }

  if (childState_ != nullptr) {
    // Have a child state machine, delegate to it.
    const Status status = childState_->onUpdate(event);
    if (status == Status::kNotDone) {
      return Status::kNotDone;
    }

    CHAR_VLOG(1) << "Child state machine is done";
    const bool childStateWasIntelligenceActor = dynamic_cast<IntelligenceActor*>(childState_.get()) != nullptr;
    if (childStateWasIntelligenceActor) {
      // The only time an intelligence actor should end is when the pvp is over, which is dictated by this state machine. Upon which, this state machine will reset the child state machine.
      throw std::runtime_error("Intelligence actor should never naturally end");
    }
    const bool childStateWasResurrect = dynamic_cast<ResurrectInPlace*>(childState_.get()) != nullptr;
    const bool childStateWasSequential = dynamic_cast<SequentialStateMachines*>(childState_.get()) != nullptr;
    childState_.reset();
    if (childStateWasResurrect) {
      // We're done resurrecting.
      CHAR_VLOG(1) << "Finished resurrecting";
      if (pvpDescriptor_) {
        // Have our pvp assignment, prepare for pvp
        setPrepareForPvpStateMachine();
        return onUpdate(event);
      } else {
        // Do not have our pvp assignment. Have we already notified the manager that we're ready for our next assignment?
        if (!publishedThatWeAreReadyForAssignment_) {
          // We haven't notified the manager that we're ready for our next assignment. Do so now.
          resetAndNotifyReadyForAssignment();
        } else {
          // We have notified the manager that we're ready for our next assignment. We now just need to wait for that assignment to arrive.
        }
        return Status::kNotDone;
      }
    } else if (childStateWasSequential) {
      CHAR_VLOG(1) << "Finished preparing for PVP";
      // Finished preparing for pvp.
      bot_.eventBroker().publishEvent<event::ReadyForPvp>(bot_.selfState()->globalId);
      weAreReady_ = true;
      if (opponentIsReady_) {
        return startPvp(event);
      } else {
        // Opponent is not ready, need to wait.
        CHAR_VLOG(1) << "We're ready for PVP, but opponent is not. Waiting for opponent to be ready.";
        return Status::kNotDone;
      }
    }
    throw std::runtime_error("PvpManager child state machine finished but reached a point which we expect to be unreachable");
  }

  if (event != nullptr) {
    if (const auto *beginPvpEvent = dynamic_cast<const event::BeginPvp*>(event); beginPvpEvent != nullptr) {
      CHAR_VLOG(1) << "Received BeginPvp";
      const common::PvpDescriptor &pvpDescriptor = beginPvpEvent->pvpDescriptor;
      if (pvpDescriptor.player1GlobalId == bot_.selfState()->globalId ||
          pvpDescriptor.player2GlobalId == bot_.selfState()->globalId) {
        // We are one of the characters in the PVP.
        if (pvpDescriptor_) {
          throw std::runtime_error("We should not have a pvp descriptor already");
        }
        pvpDescriptor_ = pvpDescriptor;
        publishedThatWeAreReadyForAssignment_ = false;
        CHAR_VLOG(1) << "We've been told to fight!";
        setPrepareForPvpStateMachine();
        return onUpdate(event);
      }
    }
  }

  // We are probably here because we are handling an event which is irrelevant for us.
  return Status::kNotDone;
}

void PvpManager::setPrepareForPvpStateMachine() {
  CHAR_VLOG(1) << "Setting state machine to prepare for PVP";
  // Start with a sequence of actions that will prepare the character for PVP.
  //  1. Move to the location of the fight.
  //  2. Make sure hp/mp are full. (if we use an hp or mp potion, we should also wait for the potion to stop healing)
  //  3. Make sure all skills are off cooldown.
  //  4. Make sure all items are off cooldown.
  //  5. Get all items necessary for the fight. TODO: Get rid of extra items.
  //  6. Repair all items.
  //  7. Enable PVP mode.
  //  8. Ensure we're visible.
  setChildStateMachine<state::machine::SequentialStateMachines>();
  state::machine::SequentialStateMachines &sequentialStateMachines = dynamic_cast<state::machine::SequentialStateMachines&>(*childState_);

  sro::Position ourPvpPosition;
  if (ourPlayer(*pvpDescriptor_, bot_.selfState()->globalId) == Player::kPlayer1) {
    ourPvpPosition = pvpDescriptor_->pvpPositionPlayer1;
  } else {
    ourPvpPosition = pvpDescriptor_->pvpPositionPlayer2;
  }

  sequentialStateMachines.emplace<state::machine::GmWarpToPosition>(ourPvpPosition);
  sequentialStateMachines.emplace<state::machine::DispelActiveBuffs>();

  // The two different players need to start from different positions.
  std::vector<packet::building::NetworkReadyPosition> waypoints = {ourPvpPosition};
  sequentialStateMachines.emplace<state::machine::Walking>(waypoints);

  sequentialStateMachines.emplace<state::machine::EnsureFullVitalsAndNoStatuses>();
  sequentialStateMachines.emplace<state::machine::GmCommandSpawnAndPickItems>(pvpDescriptor_->itemRequirements);
  sequentialStateMachines.emplace<state::machine::SpawnAndUseRepairHammerIfNecessary>();
  sequentialStateMachines.emplace<state::machine::EnablePvpMode>();
  sequentialStateMachines.emplace<state::machine::DisableGmInvisible>();
  sequentialStateMachines.emplace<state::machine::WaitForAllCooldownsToEnd>();
}

Status PvpManager::startPvp(const event::Event *event) {
  CHAR_LOG(INFO) << "Starting pvp";
  receivedResurrectionOption_ = false;
  // Both players are ready.
  rl::ai::BaseIntelligence *ourIntelligence;
  if (ourPlayer(*pvpDescriptor_, bot_.selfState()->globalId) == Player::kPlayer1) {
    ourIntelligence = pvpDescriptor_->player1Intelligence;
  } else {
    ourIntelligence = pvpDescriptor_->player2Intelligence;
  }
  CHAR_VLOG(1) << "Starting PVP with intelligence " << ourIntelligence->name();
  setChildStateMachine<state::machine::IntelligenceActor>(ourIntelligence, pvpDescriptor_->pvpId, getOpponentGlobalId());
  CHAR_VLOG(2) << "  child state machine set";
  Status result = childState_->onUpdate(event);
  CHAR_VLOG(2) << "  child state machine run, result: " << toString(result);
  return result;
}

sro::scalar_types::EntityGlobalId PvpManager::getOpponentGlobalId() {
  if (ourPlayer(*pvpDescriptor_, bot_.selfState()->globalId) == Player::kPlayer1) {
    return pvpDescriptor_->player2GlobalId;
  } else {
    return pvpDescriptor_->player1GlobalId;
  }
}

void PvpManager::resetAndNotifyReadyForAssignment() {
  CHAR_LOG(INFO) << "Resetting and notifying ready for assignment";
  weAreReady_ = false;
  opponentIsReady_ = false;
  bot_.eventBroker().publishEvent<event::PvpManagerReadyForAssignment>(bot_.sessionId());
  publishedThatWeAreReadyForAssignment_ = true;
}

} // namespace state::machine
