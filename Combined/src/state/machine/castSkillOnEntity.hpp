#ifndef STATE_MACHINE_CAST_SKILL_ON_ENTITY_HPP_
#define STATE_MACHINE_CAST_SKILL_ON_ENTITY_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <optional>

namespace state::machine {

class CastSkillOnEntity : public StateMachine {
public:
  CastSkillOnEntity(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId, sro::scalar_types::EntityGlobalId targetGlobalId, const sro::Position &positionForSkillUse);
  ~CastSkillOnEntity() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"CastSkillOnEntity"};
  const sro::scalar_types::ReferenceObjectId skillRefId_;
  const sro::scalar_types::EntityGlobalId targetGlobalId_;
  const sro::Position positionForSkillUse_;
  bool done_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_CAST_SKILL_ON_ENTITY_HPP_