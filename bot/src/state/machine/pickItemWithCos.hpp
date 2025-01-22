#ifndef STATE_MACHINE_PICK_ITEM_WITH_COS_HPP_
#define STATE_MACHINE_PICK_ITEM_WITH_COS_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

namespace state::machine {

class PickItemWithCos : public StateMachine {
public:
  PickItemWithCos(Bot &bot, sro::scalar_types::EntityGlobalId cosGlobalId, sro::scalar_types::EntityGlobalId targetGlobalId);
  ~PickItemWithCos() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"PickItemWithCos"};
  const sro::scalar_types::EntityGlobalId cosGlobalId_;
  const sro::scalar_types::EntityGlobalId targetGlobalId_;
  bool done_{false};
  bool waitingForItemToBePicked_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_PICK_ITEM_WITH_COS_HPP_