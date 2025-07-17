#include "rl/action.hpp"
#include "rl/actionSpace.hpp"

namespace rl {

std::unique_ptr<Action> ActionSpace::buildAction(state::machine::StateMachine *parentStateMachine, const sro::pk2::GameData &gameData, sro::scalar_types::EntityGlobalId opponentGlobalId, size_t actionIndex) {
  if (actionIndex == 0) {
    // IF-CHANGE: Sleep is supposed to be the first action in the action space.
    return std::make_unique<Sleep>(parentStateMachine);
  }
  actionIndex -= 1;

  if (actionIndex == 0) {
    return std::make_unique<CommonAttack>(parentStateMachine, opponentGlobalId);
  }
  actionIndex -= 1;

  if (actionIndex < kSkillIdsForObservations.size()) {
    const sro::scalar_types::ReferenceSkillId skillId = kSkillIdsForObservations[actionIndex];
    const sro::pk2::ref::Skill &skill = gameData.skillData().getSkillById(skillId);
    if (skill.targetRequired) {
      // TODO: Check if the skill is a self-targeted skill.
      return std::make_unique<TargetedSkill>(parentStateMachine, skillId, opponentGlobalId);
    } else {
      return std::make_unique<TargetlessSkill>(parentStateMachine, skillId);
    }
  }
  actionIndex -= kSkillIdsForObservations.size();

  if (actionIndex < kItemIdsForObservations.size()) {
    const sro::scalar_types::ReferenceObjectId itemId = kItemIdsForObservations[actionIndex];
    return std::make_unique<UseItem>(parentStateMachine, itemId);
  }
  throw std::runtime_error(absl::StrFormat("Action index %zu is out of bounds for action space of size %zu", actionIndex, size()));
}

} // namespace rl