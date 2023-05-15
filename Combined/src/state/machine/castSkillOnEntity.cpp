#include "castSkillOnEntity.hpp"

#include "castSkill.hpp"
#include "walking.hpp"

#include "bot.hpp"
#include "logging.hpp"

#include <silkroad_lib/position_math.h>

namespace state::machine {

CastSkillOnEntity::CastSkillOnEntity(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId, sro::scalar_types::EntityGlobalId targetGlobalId, const sro::Position &positionForSkillUse) : StateMachine(bot), skillRefId_(skillRefId), targetGlobalId_(targetGlobalId), positionForSkillUse_(positionForSkillUse) {
  stateMachineCreated(kName);
}

CastSkillOnEntity::~CastSkillOnEntity() {
  stateMachineDestroyed();
}

void CastSkillOnEntity::onUpdate(const event::Event *event) {
  // If we're walking and the target entity changes its movement, we need to abort. We don't know where to navigate to anymore and we should abort and delegate back to the parent to figure out what to do.
  if (event != nullptr) {
    if (const auto *entityMovementBegan = dynamic_cast<const event::EntityMovementBegan*>(event)) {
      if (entityMovementBegan->globalId == targetGlobalId_) {
        // The target that we're going to attack has changed their movement pattern. Interrupt the child state only if we're walking. If we're already casting a skill, there's nothing to do.
        if (childState_ && dynamic_cast<const Walking*>(childState_.get()) != nullptr) {
          if (entityMovementBeganEventsBeforeWalking_.find(event) == entityMovementBeganEventsBeforeWalking_.end()) {
            // This event came while we were walking, not before.
            // We are only walking to the target, we can safely abort.
            childState_.reset();
            done_ = true;
            return;
          }
        } else {
          // We're not walking, save this event so that if we recurse with this event while walking, we dont mistake it for a reason to abort.
          entityMovementBeganEventsBeforeWalking_.emplace(event);
        }
      }
    }
  }

  if (childState_) {
    // Have a child state, it takes priority
    childState_->onUpdate(event);
    if (childState_->done()) {
      // If we were walking, we'll need to cast the skill next.
      // If we were casting the skill, we're done.
      if (dynamic_cast<const CastSkill*>(childState_.get()) != nullptr) {
        // We were casting a skill, we are now done.
        done_ = true;
      } else if (dynamic_cast<const Walking*>(childState_.get()) != nullptr) {
        // We were walking.
        walked_ = true;
      }
      childState_.reset();
    } else {
      // Dont execute anything else in this function until the child state is done
      return;
    }
  }

  if (done_) {
    // Nothing else to do.
    return;
  }

  // No child state at this point.
  if (!walked_) {
    setChildStateMachine<Walking>(positionForSkillUse_);
    onUpdate(event);
    return;
  }

  // We are within range to cast this skill.
  auto castSkillBuilder = CastSkillStateMachineBuilder(bot_, skillRefId_).withTarget(targetGlobalId_);
  const auto &skillData = bot_.gameData().skillData().getSkillById(skillRefId_);
  const auto weaponSlot = getInventorySlotOfWeaponForSkill(skillData, bot_);
  if (weaponSlot) {
    castSkillBuilder.withWeapon(*weaponSlot);
  }
  setChildStateMachine(castSkillBuilder.create());
  onUpdate(event);
}

bool CastSkillOnEntity::done() const {
  return done_;
}

} // namespace state::machine