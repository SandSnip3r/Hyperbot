#ifndef STATE_MACHINE_TRAINING_HPP_
#define STATE_MACHINE_TRAINING_HPP_

#include "stateMachine.hpp"

#include "entity/entity.hpp"
#include "entity/geometry.hpp"

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
  bool done_{false};
  bool wantToAttackMonster(const entity::Monster &monster) const;
  void buildBuffList();
  std::optional<sro::scalar_types::ReferenceObjectId> getNextBuffToCast() const;
  std::pair<const entity::Monster*, sro::scalar_types::ReferenceObjectId> getTargetAndAttackSkill(const std::vector<const entity::Monster*> &monsters, const std::vector<sro::scalar_types::ReferenceObjectId> &attackSkills) const;
  const std::unique_ptr<entity::Geometry> trainingAreaGeometry_;
  std::vector<sro::scalar_types::ReferenceObjectId> buffsToUse_;
  std::vector<sro::scalar_types::ReferenceObjectId> skillsToUse_;
  std::set<const event::Event*> handledEvents_;
  std::optional<sro::Position> calculateWhereToWalkToAttackEntityWithSkill(const entity::MobileEntity *entity, sro::scalar_types::ReferenceObjectId attackRefId);
};

} // namespace state::machine

#endif // STATE_MACHINE_TRAINING_HPP_