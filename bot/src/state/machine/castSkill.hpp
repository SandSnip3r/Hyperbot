#ifndef STATE_MACHINE_CAST_SKILL_HPP_
#define STATE_MACHINE_CAST_SKILL_HPP_

#include "broker/eventBroker.hpp"
#include "stateMachine.hpp"

#include <silkroad_lib/pk2/ref/skill.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <optional>

namespace state::machine {

// TODO: This feels like a weird home for this function.
std::optional<uint8_t> getInventorySlotOfWeaponForSkill(const sro::pk2::ref::Skill &skillData, const Bot &bot);
std::optional<uint8_t> getInventorySlotOfShield(const Bot &bot);

class CastSkillStateMachineBuilder {
public:
  CastSkillStateMachineBuilder(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId);
  CastSkillStateMachineBuilder& withTarget(sro::scalar_types::EntityGlobalId globalId);
  CastSkillStateMachineBuilder& withWeapon(uint8_t weaponSlot);
  CastSkillStateMachineBuilder& withShield(uint8_t shieldSlot);
  CastSkillStateMachineBuilder& withImbue(sro::scalar_types::ReferenceObjectId imbueSkillRefId);
  std::unique_ptr<StateMachine> create() const;
private:
  Bot &bot_;
  sro::scalar_types::ReferenceObjectId skillRefId_;
  std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId_;
  std::optional<uint8_t> weaponSlot_;
  std::optional<uint8_t> shieldSlot_;
  std::optional<sro::scalar_types::ReferenceObjectId> imbueSkillRefId_;
};

class CastSkill : public StateMachine {
public:
// TODO: How do we do a common attack?
  CastSkill(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId, std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId, std::optional<uint8_t> weaponSlot, std::optional<uint8_t> shieldSlot, std::optional<sro::scalar_types::ReferenceObjectId> imbueSkillRefId);
  ~CastSkill() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"CastSkill"};
  static constexpr uint8_t kWeaponInventorySlot_{6};
  static constexpr uint8_t kShieldInventorySlot_{7};
  static constexpr const int kSkillCastTimeoutMs{5000};
  const sro::scalar_types::ReferenceObjectId skillRefId_;
  const std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId_;
  std::optional<uint8_t> weaponSlot_;
  std::optional<uint8_t> shieldSlot_;
  std::optional<sro::scalar_types::ReferenceObjectId> imbueSkillRefId_;
  bool expectingSkillCommandFailure_{false};
  std::optional<broker::EventBroker::EventId> skillCastTimeoutEventId_;
  bool waitingForSkillToEnd_{false};
  std::string skillName() const;
};

} // namespace state::machine

#endif // STATE_MACHINE_CAST_SKILL_HPP_