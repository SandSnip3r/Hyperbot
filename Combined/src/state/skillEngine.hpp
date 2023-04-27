#ifndef STATE_SKILL_ENGINE_HPP_
#define STATE_SKILL_ENGINE_HPP_

#include "packet/structures/packetInnerStructures.hpp"

#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <vector>

namespace state {

class SkillEngine {
public:
  void skillCooldownBegin(sro::scalar_types::ReferenceObjectId skillRefId);
  void skillCooldownEnded(sro::scalar_types::ReferenceObjectId skillRefId);
  bool skillIsOnCooldown(sro::scalar_types::ReferenceObjectId skillRefId) const;
  bool alreadyTriedToCastSkill(sro::scalar_types::ReferenceObjectId skillRefId) const;

private:
  std::set<sro::scalar_types::ReferenceObjectId> skillsOnCooldown_;

public:
  struct SkillInfo {
    SkillInfo(sro::scalar_types::EntityGlobalId a, sro::scalar_types::ReferenceObjectId b) : casterGlobalId(a), skillRefId(b) {}
    sro::scalar_types::EntityGlobalId casterGlobalId;
    sro::scalar_types::ReferenceObjectId skillRefId;
  };
  std::map<uint32_t, SkillInfo> skillCastIdMap;
  std::vector<packet::structures::ActionCommand> pendingCommandQueue;
  struct AcceptedCommandAndWasExecuted {
    AcceptedCommandAndWasExecuted(const packet::structures::ActionCommand &cmd) : command(cmd) {}
    packet::structures::ActionCommand command;
    bool wasExecuted{false};
  };
  std::vector<AcceptedCommandAndWasExecuted> acceptedCommandQueue;
};

} // namespace state

#endif // STATE_SKILL_ENGINE_HPP_