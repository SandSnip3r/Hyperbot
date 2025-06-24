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

rl::Observation buildObservation(const Bot &bot, const event::Event *event, sro::scalar_types::EntityGlobalId opponentGlobalId) {
  rl::Observation obs;
  obs.timestamp_ = std::chrono::steady_clock::now();
  obs.eventCode_ = event->eventCode;
  if (!bot.selfState()) {
    throw std::runtime_error("Cannot get an observation without a self state");
  }
  obs.ourCurrentHp_ = bot.selfState()->currentHp();
  obs.ourMaxHp_ = bot.selfState()->maxHp().value();
  obs.ourCurrentMp_ = bot.selfState()->currentMp();
  obs.ourMaxMp_ = bot.selfState()->maxMp().value();
  obs.weAreKnockedDown_ = bot.selfState()->stunnedFromKnockdown;

  std::shared_ptr<entity::Self> opponent = bot.entityTracker().getEntity<entity::Self>(opponentGlobalId);
  if (!opponent) {
    throw std::runtime_error(absl::StrFormat("Cannot find opponent with global id %d when creating observation", opponentGlobalId));
  }
  obs.opponentCurrentHp_ = opponent->currentHp();
  obs.opponentMaxHp_ = opponent->maxHp().value();
  obs.opponentCurrentMp_ = opponent->currentMp();
  obs.opponentMaxMp_ = opponent->maxMp().value();
  obs.opponentIsKnockedDown_ = opponent->stunnedFromKnockdown;

  auto countItemsInInventory = [&](sro::scalar_types::ReferenceObjectId itemId) {
    int count = 0;
    for (const auto &item : bot.selfState()->inventory) {
      if (item.refItemId == itemId) {
        count += item.getQuantity();
      }
    }
    return count;
  };
  obs.hpPotionCount_ = countItemsInInventory(5);

  static constexpr std::array kBuffs{1441, 131, 1272, 1421, 30577, 1399, 28, 554, 1410, 1398, 1335, 1271, 1380, 1256, 1253, 1377, 1281};
  if (kBuffs.size() != obs.remainingTimeOurBuffs_.size() ||
      kBuffs.size() != obs.remainingTimeOpponentBuffs_.size()) {
    throw std::runtime_error("Local buff array and observation's buff array do not have the same size");
  }
  for (int i = 0; i < kBuffs.size(); ++i) {
    const sro::scalar_types::ReferenceSkillId skillId = kBuffs[i];
    if (bot.selfState()->buffIsActive(skillId)) {
      std::optional<entity::Character::BuffData::ClockType::time_point> castTime = bot.selfState()->buffCastTime(skillId);
      if (castTime) {
        const sro::pk2::ref::Skill &skill = bot.gameData().skillData().getSkillById(skillId);
        if (skill.hasParam("DURA")) {
          obs.remainingTimeOurBuffs_[i] = std::clamp(std::chrono::duration_cast<std::chrono::milliseconds>(entity::Character::BuffData::ClockType::now()-*castTime).count() / static_cast<float>(skill.durationMs()), 0.0f, 1.0f);
        } else {
          obs.remainingTimeOurBuffs_[i] = 1.0f;
        }
      } else {
        LOG(WARNING) << "Buff " << bot.gameData().getSkillName(skillId) << " is active for us, but no cast time is known";
        obs.remainingTimeOurBuffs_[i] = 0.0f;
      }
    } else {
      obs.remainingTimeOurBuffs_[i] = 0.0f;
    }
    if (opponent->buffIsActive(skillId)) {
      std::optional<entity::Character::BuffData::ClockType::time_point> castTime = opponent->buffCastTime(skillId);
      if (castTime) {
        const sro::pk2::ref::Skill &skill = bot.gameData().skillData().getSkillById(skillId);
        if (skill.hasParam("DURA")) {
          obs.remainingTimeOpponentBuffs_[i] = std::clamp(std::chrono::duration_cast<std::chrono::milliseconds>(entity::Character::BuffData::ClockType::now()-*castTime).count() / static_cast<float>(skill.durationMs()), 0.0f, 1.0f);
        } else {
          obs.remainingTimeOpponentBuffs_[i] = 1.0f;
        }
      } else {
        LOG(WARNING) << "Buff " << bot.gameData().getSkillName(skillId) << " is active for opponent, but no cast time is known";
        obs.remainingTimeOpponentBuffs_[i] = 0.0f;
      }
    } else {
      obs.remainingTimeOpponentBuffs_[i] = 0.0f;
    }
  }

  static constexpr std::array kDebuffs{packet::enums::AbnormalStateFlag::kShocked, packet::enums::AbnormalStateFlag::kBurnt};
  static constexpr std::array kDebuffLevels{36, 41, 48, 54, 62, 82};
  if (kDebuffs.size()*kDebuffLevels.size() != obs.remainingTimeOurDebuffs_.size() ||
      kDebuffs.size()*kDebuffLevels.size() != obs.remainingTimeOpponentDebuffs_.size()) {
    throw std::runtime_error("Local debuff array and observation's debuff array do not have the same size");
  }

  const auto &ourLegacyStateEffects = bot.selfState()->legacyStateEffects();
  const auto &ourLegacyStateEndTimes = bot.selfState()->legacyStateEndTimes();
  const auto &ourLegacyStateTotalDurations = bot.selfState()->legacyStateTotalDurations();
  const auto &opponentLegacyStateEffects = opponent->legacyStateEffects();
  const auto &opponentLegacyStateEndTimes = opponent->legacyStateEndTimes();
  const auto &opponentLegacyStateTotalDurations = opponent->legacyStateTotalDurations();
  int index = 0;
  for (packet::enums::AbnormalStateFlag debuff : kDebuffs) {
    for (int level : kDebuffLevels) {
      if (ourLegacyStateEffects[helpers::toBitNum(debuff)] == level) {
        std::chrono::milliseconds remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(ourLegacyStateEndTimes[helpers::toBitNum(debuff)] - obs.timestamp_);
        obs.remainingTimeOurDebuffs_[index] = std::clamp(remainingTime.count() / static_cast<float>(ourLegacyStateTotalDurations[helpers::toBitNum(debuff)].count()), 0.0f, 1.0f);
      } else {
        obs.remainingTimeOurDebuffs_[index] = 0.0f;
      }
      if (opponentLegacyStateEffects[helpers::toBitNum(debuff)] == level) {
        std::chrono::milliseconds remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(opponentLegacyStateEndTimes[helpers::toBitNum(debuff)] - obs.timestamp_);
        obs.remainingTimeOpponentDebuffs_[index] = std::clamp(remainingTime.count() / static_cast<float>(opponentLegacyStateTotalDurations[helpers::toBitNum(debuff)].count()), 0.0f, 1.0f);
      } else {
        obs.remainingTimeOpponentDebuffs_[index] = 0.0f;
      }
      ++index;
    }
  }

  static constexpr std::array kSkills{28, 131, 554, 1253, 1256, 1271, 1272, 1281, 1335, 1377, 1380, 1398, 1399, 1410, 1421, 1441, 8312, 21209, 30577, 37, 114, 298, 300, 322, 339, 371, 588, 610, 644, 1315, 1343, 1449};
  if (kSkills.size() != obs.skillCooldowns_.size()) {
    throw std::runtime_error("Local skill array and observation's skill array do not have the same size");
  }
  for (int i = 0; i < kSkills.size(); ++i) {
    const sro::scalar_types::ReferenceSkillId skillId = kSkills[i];
    std::optional<std::chrono::milliseconds> remainingCooldown = bot.selfState()->skillRemainingCooldown(skillId);
    const int32_t totalCooldownDurationMs = bot.gameData().skillData().getSkillById(skillId).actionReuseDelay;
    obs.skillCooldowns_[i] = std::clamp(remainingCooldown.value_or(std::chrono::milliseconds(0)).count() / static_cast<float>(totalCooldownDurationMs), 0.0f, 1.0f);
  }

  static constexpr std::array kItems{5, 12, 56};
  if (kItems.size() != obs.itemCooldowns_.size()) {
    throw std::runtime_error("Local item array and observation's item array do not have the same size");
  }
  const std::map<type_id::TypeId, broker::EventBroker::EventId> &itemCooldownEventIds = bot.selfState()->getItemCooldownEventIdMap();
  for (int i = 0; i < kItems.size(); ++i) {
    const sro::scalar_types::ReferenceObjectId itemId = kItems[i];
    const int32_t totalCooldownDurationMs = [&]() {
      if (itemId == 5) {
        return bot.selfState()->getHpPotionDelay();
      } else if (itemId == 12) {
        return bot.selfState()->getMpPotionDelay();
      } else if (itemId == 56) {
        return bot.selfState()->getUniversalPillDelay();
      } else {
        throw std::runtime_error("Unknown item id: " + std::to_string(itemId));
      }
    }();
    const auto it = itemCooldownEventIds.find(itemId);
    if (it == itemCooldownEventIds.end()) {
      obs.itemCooldowns_[i] = 0.0f;
    } else {
      std::optional<std::chrono::milliseconds> remainingCooldown = bot.eventBroker().timeRemainingOnDelayedEvent(it->second);
      obs.itemCooldowns_[i] = std::clamp(remainingCooldown.value_or(std::chrono::milliseconds(0)).count() / static_cast<float>(totalCooldownDurationMs), 0.0f, 1.0f);
    }
  }

  return obs;
}

} // namespace

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
    bot_.sendActiveStateMachine();
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
    bot_.sendActiveStateMachine();
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
    if (childState_ == nullptr) {
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
    // These events which come now are for someone else's pvp. The fact that we're here now means that PvpManager has already handled ours and initiated our pvp.
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