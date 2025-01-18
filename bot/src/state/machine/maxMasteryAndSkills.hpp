#ifndef STATE_MACHINE_MAX_MASTERY_AND_SKILLS_HPP_
#define STATE_MACHINE_MAX_MASTERY_AND_SKILLS_HPP_

#include "broker/eventBroker.hpp"
#include "event/event.hpp"
#include "pk2/skillData.hpp"
#include "stateMachine.hpp"

#include <silkroad_lib/pk2/ref/mastery.h>
#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace state::machine {

namespace internal {

class SkillTree {
public:
  bool initialized() const { return initialzed_; }
  void initialize(const pk2::SkillData &skillData, sro::pk2::ref::MasteryId masteryId, uint8_t masteryLevel, const std::vector<sro::scalar_types::ReferenceSkillId> &knownSkillIds);
  bool haveSkillToLearn() const { return !skillOrdering_.empty(); }
  sro::scalar_types::ReferenceSkillId getNextSkillToLearn();
private:
  bool initialzed_{false};
  std::vector<sro::scalar_types::ReferenceSkillId> skillOrdering_;
};

} // namespace internal

class MaxMasteryAndSkills : public StateMachine {
public:
  MaxMasteryAndSkills(Bot &bot, sro::pk2::ref::MasteryId id);
  ~MaxMasteryAndSkills() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"MaxMasteryAndSkills"};
  bool done_{false};
  sro::pk2::ref::MasteryId masteryId_;
  std::optional<broker::EventBroker::EventId> timeoutEventId_;
  internal::SkillTree skillTree_;
  std::optional<sro::scalar_types::ReferenceSkillId> currentLearningSkill_;
  void resetTimeout();
};

} // namespace state::machine

#endif // STATE_MACHINE_MAX_MASTERY_AND_SKILLS_HPP_
