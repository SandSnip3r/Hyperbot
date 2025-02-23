#include "pvpManager.hpp"

#include "bot.hpp"
#include "common/sessionId.hpp"
#include "state/machine/disableGmInvisible.hpp"
#include "state/machine/enablePvpMode.hpp"
#include "state/machine/gmCommandSpawnAndPickItems.hpp"
#include "state/machine/intelligenceActor.hpp"
#include "state/machine/login.hpp"
#include "state/machine/resurrectInPlace.hpp"
#include "state/machine/spawnAndUseRepairHammerIfNecessary.hpp"
#include "state/machine/useItem.hpp"
#include "state/machine/walking.hpp"
#include "type_id/categories.hpp"

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
  LOG(INFO) << "Constructed";
}

PvpManager::~PvpManager() {
}

Status PvpManager::onUpdate(const event::Event *event) {
  if (event != nullptr) {
    if (const auto *readyForPvpEvent = dynamic_cast<const event::ReadyForPvp*>(event); readyForPvpEvent != nullptr) {
      LOG(INFO) << characterNameForLog() << "Received ReadyForPvp";
      const sro::scalar_types::EntityGlobalId opponentsGlobalId = getOpponentGlobalId();
      if (readyForPvpEvent->globalId == opponentsGlobalId) {
        opponentIsReady_ = true;
        if (weAreReady_) {
          return startPvp(event);
        }
      }
    } else if (const auto *lifeStateChanged = dynamic_cast<const event::EntityLifeStateChanged*>(event); lifeStateChanged != nullptr) {
      const bool childStateIsResurrect = dynamic_cast<ResurrectInPlace*>(childState_.get()) != nullptr;
      if (!childStateIsResurrect) {
        const bool pvping = [&]()->bool{
          if (childState_ != nullptr) {
            if (dynamic_cast<IntelligenceActor*>(childState_.get()) != nullptr) {
              return true;
            }
          }
          return false;
        }();
        if (pvping) {
          if (lifeStateChanged->globalId == bot_.selfState()->globalId ||
              lifeStateChanged->globalId == getOpponentGlobalId()) {
            // Something happened to one of us while we were pvping.
            const std::shared_ptr<entity::PlayerCharacter> playerCharacter = bot_.entityTracker().getEntity<entity::PlayerCharacter>(lifeStateChanged->globalId);
            if (playerCharacter->lifeState == sro::entity::LifeState::kDead) {
              LOG(INFO) << characterNameForLog() << "Someone died! Pvp is over.";
              childState_.reset();
              // TODO: Maybe emit a PvpEnded event so that RlTrainingManager knows.
              // If it was us, we should resurrect ourself.
              if (lifeStateChanged->globalId == bot_.selfState()->globalId) {
                LOG(INFO) << characterNameForLog() << "We were the one who died! Resurrecting in place.";
                setChildStateMachine<ResurrectInPlace>(receivedResurrectionOption_);
                return childState_->onUpdate(event);
              } else {
                // Was not us, we can immediately let the manager know that we're ready for our next assignemnt.
                LOG(INFO) << characterNameForLog() << "We weren't the one who died. Sending an event to the PvpManager that we're ready for our next assignment.";
                resetAndNotifyReadyForAssignment();
                return Status::kNotDone;
              }
            }
          }
        }
      }
    } else if (const auto *resurrectOption = dynamic_cast<const event::ResurrectOption*>(event)) {
      if (childState_ != nullptr && dynamic_cast<ResurrectInPlace*>(childState_.get()) != nullptr) {
        // Not yet resurrecting in place. We need to give the resurrect option to the ResurrectInPlace state machine.
        if (resurrectOption->option != packet::enums::ResurrectionOptionFlag::kAtPresentPoint) {
          throw std::runtime_error("We can only handle resurrecting in place");
        }
        receivedResurrectionOption_ = true;
      }
    }
  }

  if (childState_) {
    const Status status = childState_->onUpdate(event);
    if (status != Status::kDone) {
      return Status::kNotDone;
    }
    LOG(INFO) << characterNameForLog() << "Child state machine is done";
    const bool childStateWasLogin = dynamic_cast<Login*>(childState_.get()) != nullptr;
    // Ah, we're limited by we cant differentiate between different sequential state machines. Luckily for now, there's only 1.
    const bool childStateWasSequential = dynamic_cast<SequentialStateMachines*>(childState_.get()) != nullptr;
    const bool childStateWasResurrect = dynamic_cast<ResurrectInPlace*>(childState_.get()) != nullptr;
    childState_.reset();
    if (childStateWasLogin) {
      resetAndNotifyReadyForAssignment();
    } else if (childStateWasSequential) {
      LOG(INFO) << characterNameForLog() << "Finished preparing for PVP";
      // Finished preparing for pvp.
      bot_.eventBroker().publishEvent<event::ReadyForPvp>(bot_.selfState()->globalId);
      weAreReady_ = true;
      if (opponentIsReady_) {
        return startPvp(event);
      }
    } else if (childStateWasResurrect) {
      // Just finished resurrecting. Let the manager know that we're ready for our next assignemnt.
      LOG(INFO) << characterNameForLog() << "Finished resurrecting. Letting the manager know that we're ready for our next assignment.";
      resetAndNotifyReadyForAssignment();
      return Status::kNotDone;
    }
  }

  if (!bot_.loggedIn()) {
    // TODO: Improve loggedIn() function. It currently is different than what the Login state machine uses as a criteria for being logged in.
    LOG(INFO) << characterLoginInfo_.characterName << ". Need to log in";
    setChildStateMachine<state::machine::Login>(characterLoginInfo_);
    return childState_->onUpdate(event);
  }

  // Logged in
  if (event != nullptr) {
    if (const auto *beginPvpEvent = dynamic_cast<const event::BeginPvp*>(event); beginPvpEvent != nullptr) {
      LOG(INFO) << characterNameForLog() << "Received BeginPvp";
      const common::PvpDescriptor &pvpDescriptor = beginPvpEvent->pvpDescriptor;
      if (pvpDescriptor.player1GlobalId == bot_.selfState()->globalId ||
          pvpDescriptor.player2GlobalId == bot_.selfState()->globalId) {
        // We are one of the characters in the PVP.
        LOG(INFO) << characterNameForLog() << "We've been told to fight!";
        return initiatePvp(*beginPvpEvent);
      }
    }
  }

  return Status::kNotDone;
}

Status PvpManager::initiatePvp(const event::BeginPvp &beginPvpEvent) {
  pvpDescriptor_ = beginPvpEvent.pvpDescriptor;
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

  // The two different players need to start from different positions.
  std::vector<packet::building::NetworkReadyPosition> waypoints;
  if (ourPlayer(pvpDescriptor_, bot_.selfState()->globalId) == Player::kPlayer1) {
    waypoints.emplace_back(pvpDescriptor_.pvpPositionPlayer1);
  } else {
    waypoints.emplace_back(pvpDescriptor_.pvpPositionPlayer2);
  }
  sequentialStateMachines.emplace<state::machine::Walking>(waypoints);

  // Spawn and pick a single XL Vigor potion
  // Spawn and pick a single Special Universal Pill (medium)
  const sro::pk2::ref::ItemId xlVigorPotionItemId = bot_.gameData().itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kVigorPotion.contains(type_id::getTypeId(item)) && item.itemClass == 5;
  });
  const sro::pk2::ref::ItemId mediumSpecialUniversalPillItemId = bot_.gameData().itemData().getItemId([](const sro::pk2::ref::Item &item) {
    return type_id::categories::kUniversalPill.contains(type_id::getTypeId(item)) && item.itemClass == 5;
  });
  std::vector<common::ItemRequirement> fullVitalsItemRequirements = {
    {xlVigorPotionItemId, 1},
    {mediumSpecialUniversalPillItemId, 1}
  };
  sequentialStateMachines.emplace<state::machine::GmCommandSpawnAndPickItems>(fullVitalsItemRequirements);
  // Use the XL Vigor potion
  sequentialStateMachines.emplace<state::machine::UseItem>(xlVigorPotionItemId);
  // Use the Special Universal Pill (medium)
  sequentialStateMachines.emplace<state::machine::UseItem>(mediumSpecialUniversalPillItemId);

  sequentialStateMachines.emplace<state::machine::GmCommandSpawnAndPickItems>(pvpDescriptor_.itemRequirements);
  sequentialStateMachines.emplace<state::machine::SpawnAndUseRepairHammerIfNecessary>();
  sequentialStateMachines.emplace<state::machine::EnablePvpMode>();
  sequentialStateMachines.emplace<state::machine::DisableGmInvisible>();
  return sequentialStateMachines.onUpdate(&beginPvpEvent);
}

Status PvpManager::startPvp(const event::Event *event) {
  LOG(INFO) << characterNameForLog() << "Starting pvp";
  receivedResurrectionOption_ = false;
  // Both players are ready.
  setChildStateMachine<state::machine::IntelligenceActor>(getOpponentGlobalId());
  return childState_->onUpdate(event);
}

sro::scalar_types::EntityGlobalId PvpManager::getOpponentGlobalId() {
  if (ourPlayer(pvpDescriptor_, bot_.selfState()->globalId) == Player::kPlayer1) {
    return pvpDescriptor_.player2GlobalId;
  } else {
    return pvpDescriptor_.player1GlobalId;
  }
}

void PvpManager::resetAndNotifyReadyForAssignment() {
  weAreReady_ = false;
  opponentIsReady_ = false;
  bot_.eventBroker().publishEvent<event::PvpManagerReadyForAssignment>(bot_.sessionId());
}

} // namespace state::machine
