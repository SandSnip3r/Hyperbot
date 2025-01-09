#ifndef STATE_MACHINE_SELLING_ITEMS_HPP_
#define STATE_MACHINE_SELLING_ITEMS_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.h>

#include <vector>

namespace state::machine {

class SellingItems : public StateMachine {
public:
  SellingItems(Bot &bot, const std::vector<sro::scalar_types::StorageIndexType> &slotsToSell);
  ~SellingItems() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"SellingItems"};
  std::vector<sro::scalar_types::StorageIndexType> slotsToSell_;
  size_t nextToSellIndex_{0};
  bool waitingOnASell_{false};
  bool done_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_SELLING_ITEMS_HPP_