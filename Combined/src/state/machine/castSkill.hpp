#ifndef STATE_MACHINE_CAST_SKILL_HPP_
#define STATE_MACHINE_CAST_SKILL_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.h>

#include <memory>
#include <optional>

namespace state::machine {

class CastSkillStateMachineBuilder {
public:
  CastSkillStateMachineBuilder(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId);
  CastSkillStateMachineBuilder& withTarget(sro::scalar_types::EntityGlobalId globalId);
  CastSkillStateMachineBuilder& withWeapon(uint8_t weaponSlot);
  CastSkillStateMachineBuilder& withShield(uint8_t shieldSlot);
  std::unique_ptr<StateMachine> create() const;
private:
  Bot &bot_;
  sro::scalar_types::ReferenceObjectId skillRefId_;
  std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId_;
  std::optional<uint8_t> weaponSlot_;
  std::optional<uint8_t> shieldSlot_;
};

class CastSkill : public StateMachine {
public:
// TODO: How do we do a common attack?
  CastSkill(Bot &bot, sro::scalar_types::ReferenceObjectId skillRefId, std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId, std::optional<uint8_t> weaponSlot, std::optional<uint8_t> shieldSlot);
  ~CastSkill() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"CastSkill"};
  static constexpr uint8_t kWeaponInventorySlot_{6};
  static constexpr uint8_t kShieldInventorySlot_{7};
  const sro::scalar_types::ReferenceObjectId skillRefId_;
  const std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId_;
  std::optional<uint8_t> weaponSlot_;
  std::optional<uint8_t> shieldSlot_;
  std::unique_ptr<StateMachine> childState_;
  bool waitingForSkillToCast_{false};
  bool waitingForSkillToEnd_{false};
  bool done_{false};
  static constexpr int kMaxFails_{5};
  int failCount_{0};
};

} // namespace state::machine

#endif // STATE_MACHINE_CAST_SKILL_HPP_