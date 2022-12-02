#ifndef STATE_MACHINE_AUTO_POTION_HPP_
#define STATE_MACHINE_AUTO_POTION_HPP_

#include "event/event.hpp"
#include "stateMachine.hpp"
#include "type_id/typeCategory.hpp"

namespace state::machine {

// TODO: Move to a better file
enum class PotionType {
  kHp,
  kMp,
  kVigor
};

class AutoPotion : public StateMachine {
public:
  AutoPotion(Bot &bot);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  //******************************************************************************************
  //***************************************Configuration**************************************
  //******************************************************************************************
  // TODO: Move to a real configuration object
  // Potion configuration
  const double kHpThreshold_{0.90};
  const double kMpThreshold_{0.80};
  const double kVigorThreshold_{0.40};
  //******************************************************************************************
  void checkIfNeedToHeal();
  bool alreadyUsedPotion(PotionType potionType);
  void usePotion(PotionType potionType);
  void checkIfNeedToUsePill();
  bool alreadyUsedUniversalPill();
  bool alreadyUsedPurificationPill();
  void useUniversalPill();
  void usePurificationPill();
  void useItem(uint8_t slotNum, type_id::TypeId typeData);
  void handleItemWaitForReuseDelay(const event::ItemWaitForReuseDelay &event);
};

} // namespace state::machine

#endif // STATE_MACHINE_AUTO_POTION_HPP_