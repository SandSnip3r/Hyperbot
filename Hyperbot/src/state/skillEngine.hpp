#ifndef STATE_SKILL_ENGINE_HPP_
#define STATE_SKILL_ENGINE_HPP_

#include "broker/eventBroker.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include <silkroad_lib/scalar_types.h>

#include <absl/container/flat_hash_map.h>

#include <cstdint>
#include <vector>

namespace state {

class SkillEngine {
public:
  void skillCooldownBegin(sro::scalar_types::ReferenceObjectId skillRefId, broker::EventBroker::EventId cooldownEndEventId);
  void skillCooldownEnded(sro::scalar_types::ReferenceObjectId skillRefId);
  bool skillIsOnCooldown(sro::scalar_types::ReferenceObjectId skillRefId) const;
  std::optional<std::chrono::milliseconds> skillRemainingCooldown(sro::scalar_types::ReferenceObjectId skillRefId, const broker::EventBroker &eventBroker) const;
  bool alreadyTriedToCastSkill(sro::scalar_types::ReferenceObjectId skillRefId) const;
  void reset();
  void cancelEvents(broker::EventBroker &eventBroker);

private:
  absl::flat_hash_map<sro::scalar_types::ReferenceObjectId, broker::EventBroker::EventId> skillCooldownEventIdMap_;

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