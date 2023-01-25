#ifndef STATE_MACHINE_AUTO_POTION_HPP_
#define STATE_MACHINE_AUTO_POTION_HPP_

#include "stateMachine.hpp"
#include "type_id/typeCategory.hpp"

#include <silkroad_lib/scalar_types.h>

#include <memory>

namespace state::machine {

class AutoPotion : public StateMachine {
public:
  AutoPotion(Bot &bot);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  // TODO: Move to a real configuration object
  // Potion configuration
  const double kHpThreshold_{0.90};
  const double kMpThreshold_{0.90};
  const double kVigorHpThreshold_{0.45};
  const double kVigorMpThreshold_{0.30};

  std::unique_ptr<StateMachine> childState_;

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