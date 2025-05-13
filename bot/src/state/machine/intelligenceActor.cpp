#include "intelligenceActor.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "rl/action.hpp"
#include "rl/actionBuilder.hpp"
#include "rl/observation.hpp"
#include "rl/trainingManager.hpp"
#include "type_id/categories.hpp"

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>

namespace state::machine {

namespace {

rl::Observation buildObservation(const Bot &bot, const event::Event *event, sro::scalar_types::ReferenceObjectId opponentGlobalId) {
  return {bot, event, opponentGlobalId};
}

} // namespace

IntelligenceActor::IntelligenceActor(StateMachine *parent, rl::ai::BaseIntelligence *intelligence, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId) : StateMachine(parent), intelligence_(intelligence), pvpId_(pvpId), opponentGlobalId_(opponentGlobalId) {
  VLOG(1) << "Constructed " << intelligence->name() << " intelligence actor!";
  intelligence->resetForNewEpisode();
}

IntelligenceActor::~IntelligenceActor() {
}

Status IntelligenceActor::onUpdate(const event::Event *event) {
  ZoneScopedN("IntelligenceActor::onUpdate");
  if (childState_ != nullptr) {
    // The child state machine didn't immediately finish.
    // Run the update.
    const Status status = childState_->onUpdate(event);
    if (status == Status::kNotDone) {
      // Child state is not done, nothing to do for now.
      return status;
    }
    // Child state is done, reset it then continue to get our next action.
    childState_.reset();
  }

  if (event != nullptr) {
    // There are some events which we want to filter out, as they are not relevant to us.
    // TODO: Filter more.
    if (const event::CommandError *commandError = dynamic_cast<const event::CommandError*>(event); commandError != nullptr) {
      if (commandError->issuingGlobalId != bot_.selfState()->globalId) {
        // Is someone else's command error, not relevant to us.
        CHAR_VLOG(2) << "Command error from " << commandError->issuingGlobalId << ", not relevant to us";
        return Status::kNotDone;
      }
    } else if (event->eventCode == event::EventCode::kEntityMovementTimerEnded) {
      // This is an internal event, even for our own entity. If something movement related results of it, another event will come.
      CHAR_VLOG(2) << "Entity movement timer ended, not relevant to us";
      return Status::kNotDone;
    } else if (const event::ItemUseFailed *itemUseFailed = dynamic_cast<const event::ItemUseFailed*>(event); itemUseFailed != nullptr) {
      if (itemUseFailed->globalId != bot_.selfState()->globalId) {
        // Is someone else's item use failed, not relevant to us.
        CHAR_VLOG(2) << "Item use failed for " << itemUseFailed->globalId << ", not relevant to us";
        return Status::kNotDone;
      }
    } else if (event->eventCode == event::EventCode::kTimeout) {
      // Currently, agents do not create timeout events. They are typically used by tradition state machines.
      CHAR_VLOG(2) << "Timeout event, not relevant to us";
      return Status::kNotDone;
    } else if (const event::SkillCooldownEnded *skillCooldownEnded = dynamic_cast<const event::SkillCooldownEnded*>(event); skillCooldownEnded != nullptr) {
      if (skillCooldownEnded->globalId != bot_.selfState()->globalId) {
        // We do not want to have visibility into other agents' skill cooldowns.
        CHAR_VLOG(2) << "Skill cooldown ended for " << skillCooldownEnded->globalId << ", not relevant to us";
        return Status::kNotDone;
      }
    } else if (const event::ItemUseSuccess *itemUseSuccess = dynamic_cast<const event::ItemUseSuccess*>(event); itemUseSuccess != nullptr) {
      if (itemUseSuccess->globalId != bot_.selfState()->globalId) {
        // Is someone else's item use success, not relevant to us.
        CHAR_VLOG(2) << "Item use success for " << itemUseSuccess->globalId << ", not relevant to us";
        return Status::kNotDone;
      }
    } else if (const event::ItemCooldownEnded *itemCooldownEnded = dynamic_cast<const event::ItemCooldownEnded*>(event); itemCooldownEnded != nullptr) {
      if (itemCooldownEnded->globalId != bot_.selfState()->globalId) {
        // Is someone else's item cooldown ended, not relevant to us.
        CHAR_VLOG(2) << "Item cooldown ended for " << itemCooldownEnded->globalId << ", not relevant to us";
        return Status::kNotDone;
      }
    } else if (const event::ItemMoved *itemMoved = dynamic_cast<const event::ItemMoved*>(event); itemMoved != nullptr) {
      if (itemMoved->globalId != bot_.selfState()->globalId) {
        // Is someone else's item moved, not relevant to us.
        CHAR_VLOG(2) << "Item moved for " << itemMoved->globalId << ", not relevant to us";
        return Status::kNotDone;
      }
    } else if (event->eventCode == event::EventCode::kCommandSkipped) {
      // Commands being skipped are not useful for us.
      CHAR_VLOG(2) << "Command skipped, not useful to us";
      return Status::kNotDone;
    }
  }
  CHAR_VLOG(2) << "Event: " << event::toString(event->eventCode);

  const rl::Observation observation = buildObservation(bot_, event, opponentGlobalId_);

  // Check if this is a terminal state.
  if (event != nullptr) {
    if (const auto *lifeStateChanged = dynamic_cast<const event::EntityLifeStateChanged*>(event); lifeStateChanged != nullptr) {
      // Maybe someone died.
      const std::shared_ptr<entity::PlayerCharacter> playerCharacter = bot_.entityTracker().getEntity<entity::PlayerCharacter>(lifeStateChanged->globalId);
      if (playerCharacter->lifeState == sro::entity::LifeState::kDead) {
        // Someone died.
        if (lifeStateChanged->globalId == bot_.selfState()->globalId ||
            lifeStateChanged->globalId == opponentGlobalId_) {
          CHAR_VLOG(1) << "Either we or our opponent died! The pvp is over. " << lifeStateChanged->globalId << " died, we are " << bot_.selfState()->globalId;
          if (lifeStateChanged->globalId != bot_.selfState()->globalId) {
            CHAR_VLOG(1) << intelligence_->name() << " won!";
          }
          // Someone died, the pvp is over.
          // We will not query the intelligence for a chosen action, for obvious reasons.
          // We will report the state, so that it can be saved in the replay buffer.
          intelligence_->trainingManager().reportObservationAndAction(pvpId_, intelligence_->name(), observation, std::nullopt);
          return Status::kDone;
        }
      }
    }
  }

  // Since actions are state machines, immediately set the selected action as our current active child state machine.
  const bool canSendPacket = !lastPacketTime_.has_value() || (std::chrono::steady_clock::now() - lastPacketTime_.value() > kPacketSendCooldown);
  const int actionIndex = intelligence_->selectAction(bot_, observation, canSendPacket);
  CHAR_VLOG(2) << "Action " << actionIndex;
  intelligence_->trainingManager().reportObservationAndAction(pvpId_, intelligence_->name(), observation, actionIndex);
  setChildStateMachine(rl::ActionBuilder::buildAction(this, opponentGlobalId_, actionIndex));

  // Run one update on the child state machine to let it start.
  const Status status = childState_->onUpdate(event);
  if (status == Status::kDone) {
    // If the action immediately completes, deconstruct it.
    childState_.reset();
  }

  // We are never done.
  return Status::kNotDone;
}

void IntelligenceActor::injectPacket(const PacketContainer &packet, PacketContainer::Direction direction) {
  lastPacketTime_ = std::chrono::steady_clock::now();
  StateMachine::injectPacket(packet, direction);
}

} // namespace state::machine