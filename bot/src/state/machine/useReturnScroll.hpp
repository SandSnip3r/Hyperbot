#ifndef STATE_MACHINE_USE_RETURN_SCROLL_HPP_
#define STATE_MACHINE_USE_RETURN_SCROLL_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

namespace state::machine {

class UseReturnScroll : public StateMachine {
public:
  UseReturnScroll(Bot &bot, sro::scalar_types::StorageIndexType inventoryIndex);
  ~UseReturnScroll() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"UseReturnScroll"};
  sro::scalar_types::StorageIndexType inventoryIndex_;
  bool done_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_USE_RETURN_SCROLL_HPP_