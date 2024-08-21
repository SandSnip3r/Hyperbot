#ifndef STATE_MACHINE_AUTO_POTION_HPP_
#define STATE_MACHINE_AUTO_POTION_HPP_

#include "stateMachine.hpp"
#include "type_id/typeCategory.hpp"

#include <silkroad_lib/scalar_types.h>

namespace state::machine {

// TODO: For autopotion, calculate constants for:
//  1. Highest HP to use a potion and have the entire potion be used.
//  2. Lowest HP where a potion's effects are not wasted (overflow)
class AutoPotion : public StateMachine {
public:
  AutoPotion(Bot &bot);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  const sro::scalar_types::EntityGlobalId selfGlobalId_;
  bool done_{false};
  // These functions return `true` if an item was used, `false` otherwise.
  bool tryUsePurificationPill();
  bool tryUseHpPotion();
  bool tryUseMpPotion();
  bool tryUseUniversalPill();
  bool usePotion(const type_id::TypeCategory &potionType);

  void useItem(sro::scalar_types::StorageIndexType itemIndex);
};

} // namespace state::machine

#endif // STATE_MACHINE_AUTO_POTION_HPP_