#ifndef STATE_MACHINE_TRAINING_HPP_
#define STATE_MACHINE_TRAINING_HPP_

#include "stateMachine.hpp"

#include "entity/entity.hpp"

#include "../../../common/pk2/ref/skill.hpp"

#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <optional>
#include <vector>

namespace state::machine {

class Training : public StateMachine {
public:
  Training(Bot &bot, const sro::Position &trainingSpotCenter);
  ~Training() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  void buildBuffList();
  std::optional<sro::scalar_types::ReferenceObjectId> getNextBuffToCast() const;
  std::optional<uint8_t> getInventorySlotOfWeaponForSkill(const pk2::ref::Skill &skillData) const;
  std::pair<const entity::Monster*, sro::scalar_types::ReferenceObjectId> getTargetAndAttackSkill(const std::vector<const entity::Monster*> &monsters, const std::vector<sro::scalar_types::ReferenceObjectId> &attackSkills) const;
  static constexpr double kMonsterRange_{1000};
  static constexpr double kItemRange_{1000};
  sro::Position trainingSpotCenter_;
  std::vector<sro::scalar_types::ReferenceObjectId> buffsToUse_;
  std::vector<sro::scalar_types::ReferenceObjectId> skillsToUse_;
  bool waitingForSkillToCast_{false};
  bool waitingForSkillToEnd_{false};
  std::unique_ptr<StateMachine> childState_;
};

} // namespace state::machine

#endif // STATE_MACHINE_TRAINING_HPP_