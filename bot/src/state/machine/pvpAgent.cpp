#include "pvpAgent.hpp"

#include "bot.hpp"
#include "common/sessionId.hpp"
#include "state/machine/disableGmInvisible.hpp"
#include "state/machine/enablePvpMode.hpp"
#include "state/machine/gmCommandSpawnAndPickItems.hpp"
#include "state/machine/intelligenceActor.hpp"
#include "state/machine/login.hpp"
#include "state/machine/spawnAndUseRepairHammerIfNecessary.hpp"
#include "state/machine/walking.hpp"

#include <absl/log/log.h>

namespace state::machine {

namespace {

enum class Player { kPlayer1, kPlayer2 };

Player ourPlayer(const common::PvpDescriptor &pvpDescriptor, const sro::scalar_types::EntityGlobalId globalId) {
  VLOG(1) << "Looking at " << pvpDescriptor.player1GlobalId << ", " << pvpDescriptor.player2GlobalId << ", and " << globalId;
  if (pvpDescriptor.player1GlobalId == globalId) {
    return Player::kPlayer1;
  }
  if (pvpDescriptor.player2GlobalId == globalId) {
    return Player::kPlayer2;
  }
  throw std::runtime_error("We are not one of the players in the PVP");
}

} // namespace

PvpAgent::PvpAgent(Bot &bot, const CharacterLoginInfo &characterLoginInfo) : StateMachine(bot), characterLoginInfo_(characterLoginInfo) {
  LOG(INFO) << "Constructed";
}

PvpAgent::~PvpAgent() {
}

Status PvpAgent::onUpdate(const event::Event *event) {
  if (event != nullptr) {
    if (const auto *readyForPvpEvent = dynamic_cast<const event::ReadyForPvp*>(event); readyForPvpEvent != nullptr) {
      LOG(INFO) << "Received ReadyForPvp";
      const sro::scalar_types::EntityGlobalId opponentsGlobalId = getOpponentGlobalId();
      if (readyForPvpEvent->globalId == opponentsGlobalId) {
        opponentIsReady_ = true;
        if (weAreReady_) {
          return startPvp(event);
        }
      }
    }
  }

  if (childState_) {
    const Status status = childState_->onUpdate(event);
    if (status != Status::kDone) {
      return Status::kNotDone;
    }
    LOG(INFO) << "Child state machine is done";
    const bool childStateIsLogin = dynamic_cast<Login*>(childState_.get()) != nullptr;
    // Ah, we're limited by we cant differentiate between different sequential state machines. Luckily for now, there's only 1.
    const bool childStateIsSequential = dynamic_cast<SequentialStateMachines*>(childState_.get()) != nullptr;
    childState_.reset();
    if (childStateIsLogin) {
      bot_.eventBroker().publishEvent<event::PvpAgentReadyForAssignment>(bot_.sessionId());
    } else if (childStateIsSequential) {
      // Finished preparing for pvp.
      bot_.eventBroker().publishEvent<event::ReadyForPvp>(bot_.selfState()->globalId);
      weAreReady_ = true;
      if (opponentIsReady_) {
        return startPvp(event);
      }
    }
  }

  if (!bot_.loggedIn()) {
    // TODO: Improve loggedIn() function. It currently is different than what the Login state machine uses as a criteria for being logged in.
    LOG(INFO) << "Need to log in";
    setChildStateMachine<state::machine::Login>(characterLoginInfo_);
    return childState_->onUpdate(event);
  }

  // Logged in
  if (event != nullptr) {
    if (const auto *beginPvpEvent = dynamic_cast<const event::BeginPvp*>(event); beginPvpEvent != nullptr) {
      LOG(INFO) << "Received BeginPvp";
      const common::PvpDescriptor &pvpDescriptor = beginPvpEvent->pvpDescriptor;
      if (pvpDescriptor.player1GlobalId == bot_.selfState()->globalId ||
          pvpDescriptor.player2GlobalId == bot_.selfState()->globalId) {
        // We are one of the characters in the PVP.
        LOG(INFO) << "We've been told to fight!";
        return initiatePvp(*beginPvpEvent);
      }
    }
  }

  return Status::kNotDone;
}

Status PvpAgent::initiatePvp(const event::BeginPvp &beginPvpEvent) {
  pvpDescriptor_ = beginPvpEvent.pvpDescriptor;
  // Start with a sequence of actions that will prepare the character for PVP.
  //  1. Move to the location of the fight.
  //  2. Get all items necessary for the fight.
  //  3. Repair all items.
  //  4. Enable PVP mode.
  //  5. Ensure we're visible.
  setChildStateMachine<state::machine::SequentialStateMachines>();
  state::machine::SequentialStateMachines &sequentialStateMachines = dynamic_cast<state::machine::SequentialStateMachines&>(*childState_);
  std::vector<packet::building::NetworkReadyPosition> waypoints;
  if (ourPlayer(pvpDescriptor_, bot_.selfState()->globalId) == Player::kPlayer1) {
    waypoints.emplace_back(pvpDescriptor_.pvpPositionPlayer1);
  } else {
    waypoints.emplace_back(pvpDescriptor_.pvpPositionPlayer2);
  }
  sequentialStateMachines.emplace<state::machine::Walking>(waypoints);
  sequentialStateMachines.emplace<state::machine::GmCommandSpawnAndPickItems>(pvpDescriptor_.itemRequirements);
  sequentialStateMachines.emplace<state::machine::SpawnAndUseRepairHammerIfNecessary>();
  sequentialStateMachines.emplace<state::machine::EnablePvpMode>();
  sequentialStateMachines.emplace<state::machine::DisableGmInvisible>();
  return sequentialStateMachines.onUpdate(&beginPvpEvent);
}

Status PvpAgent::startPvp(const event::Event *event) {
  // Both players are ready.
  setChildStateMachine<state::machine::IntelligenceActor>(getOpponentGlobalId());
  return childState_->onUpdate(event);
}

sro::scalar_types::EntityGlobalId PvpAgent::getOpponentGlobalId() {
  if (ourPlayer(pvpDescriptor_, bot_.selfState()->globalId) == Player::kPlayer1) {
    return pvpDescriptor_.player2GlobalId;
  } else {
    return pvpDescriptor_.player1GlobalId;
  }
}

} // namespace state::machine
