#include "pvpManager.hpp"

#include "bot.hpp"
#include "common/sessionId.hpp"
#include "common/pvpDescriptor.hpp"
#include "packet/building/serverAgentEntityUpdateHwanLevel.hpp"
#include "state/machine/disableGmInvisible.hpp"
#include "state/machine/dispelActiveBuffs.hpp"
#include "state/machine/enablePvpMode.hpp"
#include "state/machine/ensureFullVitalsAndNoStatuses.hpp"
#include "state/machine/equipItem.hpp"
#include "state/machine/gmCommandSpawnAndPickItems.hpp"
#include "state/machine/gmWarpToPosition.hpp"
#include "state/machine/intelligenceActor.hpp"
#include "state/machine/login.hpp"
#include "state/machine/maybeRemoveAvatarEquipment.hpp"
#include "state/machine/resurrectInPlace.hpp"
#include "state/machine/spawnAndUseRepairHammerIfNecessary.hpp"
#include "state/machine/useItem.hpp"
#include "state/machine/waitForAllCooldownsToEnd.hpp"
#include "state/machine/walking.hpp"
#include "type_id/categories.hpp"

#include <silkroad_lib/game_constants.hpp>

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>

#include <stdexcept>

namespace state::machine {

namespace {

enum class Player { kPlayer1, kPlayer2 };

Player ourPlayer(const common::PvpDescriptor &pvpDescriptor, const std::string_view name) {
  if (pvpDescriptor.player1Name == name) {
    return Player::kPlayer1;
  }
  if (pvpDescriptor.player2Name == name) {
    return Player::kPlayer2;
  }
  throw std::runtime_error("We are not one of the players in the PVP");
}

} // namespace

PvpManager::PvpManager(Bot &bot) : StateMachine(bot) {
  LOG(INFO) << "PvpManager Constructed";
}

PvpManager::~PvpManager() {
}

Status PvpManager::onUpdate(const event::Event *event) {
  ZoneScopedN("PvpManager::onUpdate");
  if (!initialized_) {
    resetAndNotifyReadyForAssignment();
    initialized_ = true;
    return Status::kNotDone;
  }

  // Before maybe delegating to our child state machine:
  // - Check to see if someone despawned.
  // - Check to see if someone died.
  // - Check to see if our opponent is ready for pvp.
  // - Check to see if we received a resurrect option.
  if (event != nullptr) {
    if (const auto *entityDespawned = dynamic_cast<const event::EntityDespawned*>(event); entityDespawned != nullptr) {
      if (isPvping()) {
        if (!opponentGlobalId_) {
          throw std::runtime_error("We are pvping but don't have an opponent global id");
        } else if (opponentGlobalId_ && entityDespawned->globalId == *opponentGlobalId_) {
          throw std::runtime_error("Our opponent despawned during pvp");
        } else if (!bot_.selfState()) {
          throw std::runtime_error("We despawned during pvp");
        }
      }
      if (opponentGlobalId_ && entityDespawned->globalId == *opponentGlobalId_) {
        CHAR_LOG(INFO) << "Our opponent despawned!";
        opponentGlobalId_.reset();
      }
    } else if (const auto *entitySpawned = dynamic_cast<const event::EntitySpawned*>(event); entitySpawned != nullptr) {
      if (pvpDescriptor_ && !opponentGlobalId_) {
        // We don't have an opponent yet. Check if this is our opponent.
        const std::shared_ptr<entity::PlayerCharacter> playerCharacter = bot_.entityTracker().getEntity<entity::PlayerCharacter>(entitySpawned->globalId);
        if (playerCharacter && playerCharacter->name == pvpDescriptor_->player2Name) {
          opponentGlobalId_ = entitySpawned->globalId;
          CHAR_LOG(INFO) << "Found our opponent " << *opponentGlobalId_ << "!";
        }
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
          if (isPvping()) {
            if (!pvpDescriptor_) {
              throw std::runtime_error("We are pvping but don't have a pvp descriptor");
            }
            if (playerCharacter->name == pvpDescriptor_->player1Name ||
                playerCharacter->name == pvpDescriptor_->player2Name) {
              CHAR_LOG(INFO) << "Either we or our opponent died! The pvp is over. " << playerCharacter->name << " died, we are " << bot_.selfState()->globalId;
              // Call the child state once more, so that it may report the final state to the replay buffer.
              Status childStatus = childState_->onUpdate(event);
              (void)childStatus; // We do not care what the child state returns because we are going to reset it regardless.
              childState_.reset();
              bot_.sendActiveStateMachine();
              pvpDescriptor_.reset();
              // If it was the opponent who died, we can immediately report that we're ready for our next assignment.
              if (!opponentGlobalId_) {
                throw std::runtime_error("Someone died while we were pvping but we have no opponent");
              }
              if (lifeStateChanged->globalId == *opponentGlobalId_) {
                resetAndNotifyReadyForAssignment();
                return Status::kNotDone;
              } else {
                CHAR_VLOG(1) << "We died while pvping";
              }
              if (childStatus != Status::kDone) {
                // Sent a terminal state to the child state machine, it should have returned "Done".
                throw std::runtime_error("Child state machine did not return Done after receiving a terminal state");
              }
            }
          }
          if (lifeStateChanged->globalId == bot_.selfState()->globalId) {
            // If we died, we need to resurrect ourself, regardless of what we're doing.
            CHAR_VLOG(1) << "We died, resurrecting";
            setChildStateMachine<ResurrectInPlace>(receivedResurrectionOption_);
            return onUpdate(event);
          }
        }
      }
    } else if (const auto *readyForPvpEvent = dynamic_cast<const event::ReadyForPvp*>(event); readyForPvpEvent != nullptr) {
      if (opponentGlobalId_) {
        const sro::scalar_types::EntityGlobalId opponentsGlobalId = *opponentGlobalId_;
        if (readyForPvpEvent->globalId == opponentsGlobalId) {
          CHAR_VLOG(1) << "Received ReadyForPvp from our opponent";
          opponentIsReady_ = true;
          if (weAreReady_) {
            return startPvp(event);
          } else {
            CHAR_VLOG(1) << "Opponent is ready for PVP, but we are not. Waiting for us to be ready.";
            return Status::kNotDone;
          }
        }
      }
    } else if (const auto *resurrectOption = dynamic_cast<const event::ResurrectOption*>(event)) {
      if (resurrectOption->globalId == bot_.selfState()->globalId) {
        if (childState_ != nullptr && dynamic_cast<ResurrectInPlace*>(childState_.get()) != nullptr) {
          // Not yet resurrecting in place. Take note that we received this so that when the ResurrectInPlace state machine is run, it knows not to wait for this.
          // We need to do this because the (resurrect options) and (died) packets can come in any order.
          if (resurrectOption->option != packet::enums::ResurrectionOptionFlag::kAtPresentPoint) {
            throw std::runtime_error("We can only handle resurrecting in place");
          }
          receivedResurrectionOption_ = true;
        }
      }
    } else if (const auto *beginPvpEvent = dynamic_cast<const event::BeginPvp*>(event); beginPvpEvent != nullptr) {
      CHAR_VLOG(1) << "Received BeginPvp";
      if (bot_.selfState()) {
        const common::PvpDescriptor &pvpDescriptor = beginPvpEvent->pvpDescriptor;
        if (pvpDescriptor.player1Name == bot_.selfState()->name ||
            pvpDescriptor.player2Name == bot_.selfState()->name) {
          // We are one of the characters in the PVP.
          if (pvpDescriptor_) {
            throw std::runtime_error(absl::StrFormat("PvpManager: [%s] Received pvp descriptor for us but we already have one", bot_.selfState()->name));
          }
          pvpDescriptor_ = pvpDescriptor;
          // Since the pvp descriptor refers to our opponent by name, we will separately track the global id of our opponent.
          const std::string_view opponentName = pvpDescriptor_->player1Name == bot_.selfState()->name ? pvpDescriptor_->player2Name : pvpDescriptor_->player1Name;
          CHAR_LOG(INFO) << "Preparing to fight against " << opponentName;
          std::shared_ptr<entity::PlayerCharacter> opponent = bot_.entityTracker().getPlayerByName(opponentName);
          if (!opponent) {
            throw std::runtime_error("We don't have an opponent");
          }
          opponentGlobalId_ = opponent->globalId;
          publishedThatWeAreReadyForAssignment_ = false;
          CHAR_VLOG(1) << "We've been told to fight against " << opponentName << "!";
          if (childState_ == nullptr) {
            setPrepareForPvpStateMachine();
            return onUpdate(nullptr);
          } else {
            CHAR_VLOG(1) << "We already have a child state machine. We will wait until that finishes before preparing for pvp";
          }
        }
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
    const bool childStateWasSequential =
        dynamic_cast<SequentialStateMachines*>(childState_.get()) != nullptr;
    childState_.reset();
    bot_.sendActiveStateMachine();
    if (childStateWasResurrect) {
      // We're done resurrecting.
      CHAR_VLOG(1) << "Finished resurrecting";
      if (pvpDescriptor_) {
        CHAR_VLOG(1) << "Have our pvp assignment, preparing for pvp";
        // Have our pvp assignment, prepare for pvp
        setPrepareForPvpStateMachine();
        return onUpdate(event);
      } else {
        // Do not have our pvp assignment. Have we already notified the manager that we're ready for our next assignment?
        if (!publishedThatWeAreReadyForAssignment_) {
          // We haven't notified the manager that we're ready for our next assignment. Do so now.
          CHAR_VLOG(1) << "Do not have a pvp assignment. We haven't notified the manager that we're ready for our next assignment. Do so now.";
          resetAndNotifyReadyForAssignment();
        } else {
          // We have notified the manager that we're ready for our next assignment. We now just need to wait for that assignment to arrive.
          CHAR_VLOG(1) << "Do not have a pvp assignment. We have already notified the manager that we're ready for our next assignment. We now just need to wait for that assignment to arrive.";
        }
        return Status::kNotDone;
      }
    } else if (childStateWasSequential) {
      CHAR_VLOG(1) << "Finished preparing for PVP, publishing ReadyForPvp event";
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

  // We are probably here because we are handling an event which is irrelevant for us.
  return Status::kNotDone;
}

void PvpManager::setPrepareForPvpStateMachine() {
  sro::scalar_types::ReferenceObjectId deepLearningHatRefId = 23958; // Wizard's Hat (M)
  sro::scalar_types::ReferenceObjectId randomHatRefId = 24302; // Joker Hat (M)

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

  // Take off any avatar hats we may be wearing.
  sequentialStateMachines.emplace<state::machine::MaybeRemoveAvatarEquipment>(
      /*slot=*/sro::game_constants::kAvatarHatSlot,
      /*targetSlot=*/[](Bot &bot) {
        // Find the last free slot in our inventory.
        for (int i=bot.inventory().size()-1; i>=0; --i) {
          if (!bot.inventory().hasItem(i)) {
            return i;
          }
        }
        throw std::runtime_error("We don't have any free slots in our inventory");
      });

  sro::Position ourPvpPosition;
  if (ourPlayer(*pvpDescriptor_, bot_.selfState()->name) == Player::kPlayer1) {
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

  // Equip an avatar hat depending on which intelligence we are.
  std::string ourIntelligenceName;
  if (ourPlayer(*pvpDescriptor_, bot_.selfState()->name) == Player::kPlayer1) {
    ourIntelligenceName = pvpDescriptor_->player1Intelligence->name();
  } else {
    ourIntelligenceName = pvpDescriptor_->player2Intelligence->name();
  }
  sequentialStateMachines.emplace<state::machine::EquipItem>(ourIntelligenceName == "Random" ? randomHatRefId : deepLearningHatRefId);

  sequentialStateMachines.emplace<state::machine::DisableGmInvisible>();
  sequentialStateMachines.emplace<state::machine::WaitForAllCooldownsToEnd>();
}

Status PvpManager::startPvp(const event::Event *event) {
  CHAR_LOG(INFO) << "Starting pvp";
  receivedResurrectionOption_ = false;
  // Both players are ready.
  std::shared_ptr<rl::ai::BaseIntelligence> ourIntelligence;
  if (ourPlayer(*pvpDescriptor_, bot_.selfState()->name) == Player::kPlayer1) {
    ourIntelligence = pvpDescriptor_->player1Intelligence;
  } else {
    ourIntelligence = pvpDescriptor_->player2Intelligence;
  }
  CHAR_VLOG(1) << "Starting PVP with intelligence " << ourIntelligence->name();
  if (!opponentGlobalId_) {
    throw std::runtime_error("Starting pvp but we don't have an opponent global id");
  }
  setChildStateMachine<state::machine::IntelligenceActor>(std::move(ourIntelligence), pvpDescriptor_->pvpId, *opponentGlobalId_);
  CHAR_VLOG(2) << "  child state machine set";
  Status result = childState_->onUpdate(event);
  CHAR_VLOG(2) << "  child state machine run, result: " << toString(result);
  return result;
}

void PvpManager::resetAndNotifyReadyForAssignment() {
  CHAR_LOG(INFO) << "Resetting and notifying ready for assignment";
  weAreReady_ = false;
  opponentIsReady_ = false;
  bot_.eventBroker().publishEvent<event::PvpManagerReadyForAssignment>(bot_.sessionId());
  publishedThatWeAreReadyForAssignment_ = true;
}

bool PvpManager::isPvping() const {
  if (childState_ != nullptr) {
    if (dynamic_cast<IntelligenceActor*>(childState_.get()) != nullptr) {
      return true;
    }
  }
  return false;
}

} // namespace state::machine
