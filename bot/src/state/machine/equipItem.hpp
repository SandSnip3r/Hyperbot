#ifndef STATE_MACHINE_EQUIP_ITEM_HPP_
#define STATE_MACHINE_EQUIP_ITEM_HPP_

#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <string>

namespace state::machine {

class EquipItem : public StateMachine {
public:
  EquipItem(StateMachine *parent, sro::scalar_types::ReferenceObjectId itemRefId);
  ~EquipItem() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"EquipItem"};
  const sro::scalar_types::ReferenceObjectId itemRefId_;
};

} // namespace state::machine

#endif // STATE_MACHINE_EQUIP_ITEM_HPP_
