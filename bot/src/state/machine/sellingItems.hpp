#ifndef STATE_MACHINE_SELLING_ITEMS_HPP_
#define STATE_MACHINE_SELLING_ITEMS_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <vector>

namespace state::machine {

class SellingItems : public StateMachine {
public:
  SellingItems(StateMachine *parent, const std::vector<sro::scalar_types::StorageIndexType> &slotsToSell);
  ~SellingItems() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"SellingItems"};
  std::vector<sro::scalar_types::StorageIndexType> slotsToSell_;
  size_t nextToSellIndex_{0};
  bool waitingOnASell_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_SELLING_ITEMS_HPP_