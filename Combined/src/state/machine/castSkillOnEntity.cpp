#include "castSkillOnEntity.hpp"

#include "castSkill.hpp"
#include "walking.hpp"

#include "bot.hpp"
#include "logging.hpp"

#include <silkroad_lib/position_math.h>

namespace state::machine {

CastSkillOnEntity::CastSkillOnEntity(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId, sro::scalar_types::EntityGlobalId targetGlobalId, const sro::Position &positionForSkillUse) : StateMachine(bot), skillRefId_(skillRefId), targetGlobalId_(targetGlobalId), positionForSkillUse_(positionForSkillUse) {
  stateMachineCreated(kName);

  if (sro::position_math::calculateDistance2d(bot_.selfState().position(), positionForSkillUse_) > bot_.gameData().skillData().getSkillById(skillRefId).actionRange) {
    // Not close enough to cast the skill.
    setChildStateMachine<Walking>(bot_, positionForSkillUse_);
  } else {
    // We are close enough to cast the skill from where we stand
  }
}

CastSkillOnEntity::~CastSkillOnEntity() {
  stateMachineDestroyed();
}

void CastSkillOnEntity::onUpdate(const event::Event *event) {
  // TODO: If we're walking and the target entity changes its movement, we would need to interrupt. Rather than change the navigation plan here, we should abort and delegate back to the parent to figure out what to do.

  if (childState_) {
    // Have a child state, it takes priority
    childState_->onUpdate(event);
    if (childState_->done()) {
      // If we were walking, we'll need to cast the skill next.
      // If we were casting the skill, we're done.
      if (dynamic_cast<const CastSkill*>(childState_.get()) != nullptr) {
        // We were casting a skill, we are now done.
        done_ = true;
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