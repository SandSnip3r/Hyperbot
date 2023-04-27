#ifndef STATE_MACHINE_PICK_ITEM_HPP_
#define STATE_MACHINE_PICK_ITEM_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.h>

namespace state::machine {

class PickItem : public StateMachine {
public:
  PickItem(Bot &bot, sro::scalar_types::EntityGlobalId targetGlobalId);
  ~PickItem() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"PickItem"};
  const sro::scalar_types::EntityGlobalId targetGlobalId_;
  bool done_{false};
  bool waitingForItemToBePicked_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_PICK_ITEM_HPP_