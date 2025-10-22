#include "intelligenceActor.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "rl/action.hpp"
#include "rl/actionSpace.hpp"
#include "rl/ai/deepLearningIntelligence.hpp"
#include "rl/observation.hpp"
#include "rl/observationBuilder.hpp"
#include "rl/trainingManager.hpp"
#include "type_id/categories.hpp"

#include <tracy/Tracy.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_join.h>

namespace state::machine {

IntelligenceActor::IntelligenceActor(StateMachine *parent, std::shared_ptr<rl::ai::BaseIntelligence> intelligence, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId) : StateMachine(parent), intelligence_(std::move(intelligence)), pvpId_(pvpId), opponentGlobalId_(opponentGlobalId) {
  VLOG(1) << "Constructed " << intelligence_->name() << " intelligence actor!";
}

IntelligenceActor::~IntelligenceActor() {
}

Status IntelligenceActor::onUpdate(const event::Event *event) {
  ZoneScopedN("IntelligenceActor::onUpdate");

  if (!isRelevantEvent(event)) {
    // We'll return early so that we don't act on this event.
    return Status::kNotDone;
  }

  if (haveChild()) {
    // The child state machine didn't immediately finish.
    // Run the update.
    const Status status = onUpdateChild(event);
    if (status == Status::kNotDone) {
      // Child state is not done, nothing to do for now.
      return status;
    }
    // Child state is done, reset it then continue to get our next action.
    resetChild();
  }
  CHAR_VLOG(2) << "Event: " << event::toString(event->eventCode);

  rl::Observation observation;
  rl::ObservationBuilder::buildObservationFromBot(bot_, observation, opponentGlobalId_);

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
          if (dynamic_cast<rl::ai::DeepLearningIntelligence*>(intelligence_.get()) != nullptr) {
            intelligence_->trainingManager().reportObservationAndAction(pvpId_, intelligence_->name(), observation, std::nullopt);
          }
          return Status::kDone;
        }
      }
    }
  }

  // Since actions are state machines, immediately set the selected action as our current active child state machine.
  const bool canSendPacket = !lastPacketTime_.has_value() || (std::chrono::steady_clock::now() - lastPacketTime_.value() > kPacketSendCooldown);
  const int actionIndex = intelligence_->selectAction(bot_, observation, canSendPacket, std::to_string(pvpId_));
  CHAR_VLOG(2) << "Action " << actionIndex;
  if (dynamic_cast<rl::ai::DeepLearningIntelligence*>(intelligence_.get()) != nullptr) {
    intelligence_->trainingManager().reportObservationAndAction(pvpId_, intelligence_->name(), observation, actionIndex);
  }
  setChild(rl::ActionSpace::buildAction(this, bot_.gameData(), opponentGlobalId_, actionIndex));

  // Run one update on the child state machine to let it start.
  const Status status = onUpdateChild(event);
  if (status == Status::kDone) {
    // If the action immediately completes, deconstruct it.
    resetChild();
  }

  // We are never done.
  return Status::kNotDone;
}

void IntelligenceActor::injectPacket(const PacketContainer &packet, PacketContainer::Direction direction) {
  lastPacketTime_ = std::chrono::steady_clock::now();
  StateMachine::injectPacket(packet, direction);
}

bool IntelligenceActor::isRelevantEvent(const event::Event *event) const {
  if (event == nullptr) {
    return true;
  }
  if (bot_.selfState() == nullptr) {
    // We do not have a self state, so we cannot determine if this event is relevant.
    return true;
  }
  // There are some events which we want to filter out, as they are not relevant to us.
  if (const event::CommandError *commandError = dynamic_cast<const event::CommandError*>(event); commandError != nullptr) {
    if (commandError->issuingGlobalId != bot_.selfState()->globalId) {
      // Is someone else's command error, not relevant to us.
      return false;
    }
  } else if (event->eventCode == event::EventCode::kEntityMovementTimerEnded) {
    // This is an internal event, we should not handle it even if it is for us. If something movement related results of it, another "public" event will come.
    return false;
  } else if (const event::ItemUseFailed *itemUseFailed = dynamic_cast<const event::ItemUseFailed*>(event); itemUseFailed != nullptr) {
    if (itemUseFailed->globalId != bot_.selfState()->globalId) {
      // Is someone else's item use failed, not relevant to us.
      return false;
    }
  } else if (event->eventCode == event::EventCode::kTimeout) {
    if (!haveChild()) {
      // Apart from Actions taken, which are run as child state machines, this state machine does not have any reason to see timeouts.
      return false;
    }
  } else if (const event::SkillCooldownEnded *skillCooldownEnded = dynamic_cast<const event::SkillCooldownEnded*>(event); skillCooldownEnded != nullptr) {
    if (skillCooldownEnded->globalId != bot_.selfState()->globalId) {
      // We do not want to have visibility into other agents' skill cooldowns.
      return false;
    }
  } else if (const event::ItemUseSuccess *itemUseSuccess = dynamic_cast<const event::ItemUseSuccess*>(event); itemUseSuccess != nullptr) {
    if (itemUseSuccess->globalId != bot_.selfState()->globalId) {
      // Is someone else's item use success, not relevant to us.
      return false;
    }
  } else if (const event::ItemCooldownEnded *itemCooldownEnded = dynamic_cast<const event::ItemCooldownEnded*>(event); itemCooldownEnded != nullptr) {
    if (itemCooldownEnded->globalId != bot_.selfState()->globalId) {
      // Is someone else's item cooldown ended, not relevant to us.
      return false;
    }
  } else if (const event::ItemMoved *itemMoved = dynamic_cast<const event::ItemMoved*>(event); itemMoved != nullptr) {
    if (itemMoved->globalId != bot_.selfState()->globalId) {
      // Is someone else's item moved, not relevant to us.
      return false;
    }
  } else if (event->eventCode == event::EventCode::kCommandSkipped) {
    // Commands being skipped are not useful for us.
    return false;
  } else if (event->eventCode == event::EventCode::kInternalItemCooldownEnded) {
    // This is an internal event, even for our own entity. If something item related results of it, another event will come.
    return false;
  } else if (event->eventCode == event::EventCode::kInternalSkillCooldownEnded) {
    // This is an internal event, even for our own entity. If something skill related results of it, another event will come.
    return false;
  } else if (const event::SkillFailed *skillFailed = dynamic_cast<const event::SkillFailed*>(event); skillFailed != nullptr) {
    if (skillFailed->casterGlobalId != bot_.selfState()->globalId) {
      // We do not want to have visibility into other agents' skill failures.
      return false;
    }
  } else if (const event::EntityHpChanged *entityHpChanged = dynamic_cast<const event::EntityHpChanged*>(event); entityHpChanged != nullptr) {
    if (entityHpChanged->globalId != bot_.selfState()->globalId && entityHpChanged->globalId != opponentGlobalId_) {
      // We only want to see our own and our opponent's hp changes.
      return false;
    }
  } else if (const event::EntityMpChanged *entityMpChanged = dynamic_cast<const event::EntityMpChanged*>(event); entityMpChanged != nullptr) {
    if (entityMpChanged->globalId != bot_.selfState()->globalId && entityMpChanged->globalId != opponentGlobalId_) {
      // We only want to see our own and our opponent's mp changes.
      return false;
    }
  } else if (const event::DealtDamage *dealtDamage = dynamic_cast<const event::DealtDamage*>(event); dealtDamage != nullptr) {
    if (dealtDamage->sourceId != bot_.selfState()->globalId || dealtDamage->targetId != opponentGlobalId_) {
      // We only want to see our own and our opponent's dealt damage.
      return false;
    }
  } else if (const event::SkillBegan *skillBegan = dynamic_cast<const event::SkillBegan*>(event); skillBegan != nullptr) {
    if (skillBegan->casterGlobalId != bot_.selfState()->globalId && skillBegan->casterGlobalId != opponentGlobalId_) {
      // We do not want to have visibility into other agents' skill beginnings.
      return false;
    }
  } else if (const event::SkillEnded *skillEnded = dynamic_cast<const event::SkillEnded*>(event); skillEnded != nullptr) {
    if (skillEnded->casterGlobalId != bot_.selfState()->globalId && skillEnded->casterGlobalId != opponentGlobalId_) {
      // We do not want to have visibility into other agents' skill endings.
      return false;
    }
  } else if (const event::BuffAdded *buffAdded = dynamic_cast<const event::BuffAdded*>(event); buffAdded != nullptr) {
    if (buffAdded->entityGlobalId != bot_.selfState()->globalId && buffAdded->entityGlobalId != opponentGlobalId_) {
      // We only care about buffs relevant to our PVP.
      return false;
    }
  } else if (const event::BuffRemoved *buffRemoved = dynamic_cast<const event::BuffRemoved*>(event); buffRemoved != nullptr) {
    if (buffRemoved->entityGlobalId != bot_.selfState()->globalId && buffRemoved->entityGlobalId != opponentGlobalId_) {
      // We only care about buffs relevant to our PVP.
      return false;
    }
  } else if (const event::EntityStatesChanged *entityStatesChanged = dynamic_cast<const event::EntityStatesChanged*>(event); entityStatesChanged != nullptr) {
    if (entityStatesChanged->globalId != bot_.selfState()->globalId && entityStatesChanged->globalId != opponentGlobalId_) {
      // We only care about entity states relevant to our PVP.
      return false;
    }
  } else if (event->eventCode == event::EventCode::kInventoryItemUpdated) {
    // We do not care about inventory item updated events. These are probably things like durability changed.
    return false;
  } else if (event->eventCode == event::EventCode::kEntityMovementBegan ||
             event->eventCode == event::EventCode::kEntityMovementEnded ||
             event->eventCode == event::EventCode::kEntityNotMovingAngleChanged ||
             event->eventCode == event::EventCode::kWalkingPathUpdated ||
             event->eventCode == event::EventCode::kEntityPositionUpdated) {
    // We do not care about entity movement events.
    return false;
  } else if (event->eventCode == event::EventCode::kEntityDespawned ||
             event->eventCode == event::EventCode::kEntitySpawned) {
    // We do not care about entity spawned/despawned events.
    return false;
  } else if (event->eventCode == event::EventCode::kReadyForPvp ||
             event->eventCode == event::EventCode::kPvpManagerReadyForAssignment ||
             event->eventCode == event::EventCode::kBeginPvp)  {
    // We are already PVPing (we would not be in this function if we were not). These are for someone else's PVP.
    return false;
  } else if (const event::KnockedDown *knockedDown = dynamic_cast<const event::KnockedDown*>(event); knockedDown != nullptr) {
    if (knockedDown->globalId != bot_.selfState()->globalId && knockedDown->globalId != opponentGlobalId_) {
      // We do not want to have visibility into knocked down events from other fights.
      return false;
    }
  } else if (const event::KnockdownStunEnded *knockdownStunEnded = dynamic_cast<const event::KnockdownStunEnded*>(event); knockdownStunEnded != nullptr) {
    if (knockdownStunEnded->globalId != bot_.selfState()->globalId && knockdownStunEnded->globalId != opponentGlobalId_) {
      // We do not want to have visibility into knockdown stun ended events from other fights.
      return false;
    }
  } else if (event->eventCode == event::EventCode::kResurrectOption) {
    // A parent state machine handles resurrection options. We're just here to fight.
    return false;
  }

  // We're interested in all other events.
  return true;
}

} // namespace state::machine