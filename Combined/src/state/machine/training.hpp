#ifndef STATE_MACHINE_TRAINING_HPP_
#define STATE_MACHINE_TRAINING_HPP_

#include "stateMachine.hpp"

#include "entity/entity.hpp"
#include "entity/geometry.hpp"

#include "../../../common/pk2/ref/skill.hpp"

#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace state::machine {

class Training : public StateMachine {
public:
  Training(Bot &bot, std::unique_ptr<entity::Geometry> &&trainingAreaGeometry);
  ~Training() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"Training"};
  bool wantToAttackMonster(const entity::Monster &monster) const;
  void buildBuffList();
  std::optional<sro::scalar_types::ReferenceObjectId> getNextBuffToCast() const;
  bool canCastSkill(sro::scalar_types::ReferenceObjectId skillRefId) const;
  std::optional<uint8_t> getInventorySlotOfWeaponForSkill(const pk2::ref::Skill &skillData) const;
  std::pair<const entity::Monster*, sro::scalar_types::ReferenceObjectId> getTargetAndAttackSkill(const std::vector<const entity::Monster*> &monsters, const std::vector<sro::scalar_types::ReferenceObjectId> &attackSkills) const;
  const std::unique_ptr<entity::Geometry> trainingAreaGeometry_;
  std::vector<sro::scalar_types::ReferenceObjectId> buffsToUse_;
  std::vector<sro::scalar_types::ReferenceObjectId> skillsToUse_;
  bool waitingForSkillToCast_{false};
  bool waitingForSkillToEnd_{false};
  std::unique_ptr<StateMachine> childState_;
  // Once we successfully use a skill, there is a little bit of time before the buff gets applied to us. We want to know which buffs we're expecting to eventually come so that we dont recast them too early.
  std::set<sro::scalar_types::ReferenceObjectId> buffsCastButNotYetActive_;
};

} // namespace state::machine

#endif // STATE_MACHINE_TRAINING_HPP_