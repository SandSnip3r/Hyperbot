#ifndef STATE_MACHINE_MAYBE_REMOVE_AVATAR_EQUIPMENT_HPP_
#define STATE_MACHINE_MAYBE_REMOVE_AVATAR_EQUIPMENT_HPP_

#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <functional>
#include <string>

namespace state::machine {

class MaybeRemoveAvatarEquipment : public StateMachine {
public:
  MaybeRemoveAvatarEquipment(StateMachine *parent, sro::scalar_types::StorageIndexType avatarSlot, std::function<sro::scalar_types::StorageIndexType(Bot&)> targetSlot);
  ~MaybeRemoveAvatarEquipment() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"MaybeRemoveAvatarEquipment"};
  sro::scalar_types::StorageIndexType avatarSlot_;
  std::function<sro::scalar_types::StorageIndexType(Bot&)> targetSlot_;
  sro::scalar_types::StorageIndexType intermediateSlot_;
  bool unequipping_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_MAYBE_REMOVE_AVATAR_EQUIPMENT_HPP_
