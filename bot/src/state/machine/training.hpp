#ifndef STATE_MACHINE_TRAINING_HPP_
#define STATE_MACHINE_TRAINING_HPP_

#include "stateMachine.hpp"
#include "walking.hpp"

#include "entity/entity.hpp"
#include "entity/geometry.hpp"
#include "entity/item.hpp"
#include "entity/monster.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace state::machine {

class Training : public StateMachine {
public:
  Training(Bot &bot, std::unique_ptr<entity::Geometry> &&trainingAreaGeometry);
  ~Training() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"Training"};
  bool wantToAttackMonster(const entity::Monster &monster) const;
  void getSkillsFromConfig();
  void resetSkillLists();
  using SkillList = std::vector<sro::scalar_types::ReferenceObjectId>;
  using ItemList = std::vector<const entity::Item*>;
  using MonsterList = std::vector<const entity::Monster*>;
  void removeSkillsFromListWhichWeDontHave(SkillList &skillList);
  std::optional<sro::scalar_types::ReferenceObjectId> getNextBuffToCast(const SkillList &buffList) const;

  struct TargetAndAttackSkill {
    sro::scalar_types::EntityGlobalId targetId;
    sro::scalar_types::ReferenceObjectId skillId;
  };
  std::optional<TargetAndAttackSkill> getTargetAndAttackSkill(const MonsterList &monsters) const;
  std::unique_ptr<StateMachine> applyStatPointsChildStateMachine_;
  const std::unique_ptr<entity::Geometry> trainingAreaGeometry_;
  SkillList trainingBuffs_;
  SkillList nonTrainingBuffs_;
  SkillList skillsToUse_;
  std::optional<sro::scalar_types::ReferenceObjectId> imbueRefId_;
  std::set<const event::Event*> handledEvents_;
  std::optional<TargetAndAttackSkill> walkingTargetAndAttack_;
  std::optional<sro::scalar_types::EntityGlobalId> walkingToItemTarget_;
  std::optional<sro::Position> calculateWhereToWalkToAttackEntityWithSkill(const entity::MobileEntity &entity, sro::scalar_types::ReferenceObjectId attackRefId);
  bool checkBuffs(const SkillList &buffList);
  void possiblyOverwriteChildStateMachine(std::unique_ptr<StateMachine> newChildStateMachine);

  template<typename StateMachineType, typename ...Args>
  void possiblyOverwriteChildStateMachine(Args&& ...args) {
    if (childState_ && dynamic_cast<Walking*>(childState_.get()) == nullptr) {
      throw std::runtime_error("Cannot overwrite a child state which is not Walking");
    }
    walkingTargetAndAttack_.reset();
    setChildStateMachine<StateMachineType>(std::forward<Args>(args)...);
  }

  std::tuple<ItemList, MonsterList> getItemsAndMonstersInRange() const;
  bool tryPickItem(const ItemList &itemList);
  bool tryAttackMonster(const MonsterList &monsterList);
  bool walkToRandomPoint();
};

} // namespace state::machine

#endif // STATE_MACHINE_TRAINING_HPP_