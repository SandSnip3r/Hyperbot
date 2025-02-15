#ifndef STATE_MACHINE_BUYING_ITEMS_HPP_
#define STATE_MACHINE_BUYING_ITEMS_HPP_

#include "stateMachine.hpp"

#include <cstdint>
#include <map>

namespace state::machine {

class BuyingItems : public StateMachine {
public:
  struct PurchaseRequest {
    uint8_t tabIndex;
    uint8_t itemIndex;
    uint16_t quantity;
    int32_t maxStackSize;
  };
  BuyingItems(Bot &bot, const std::map<uint32_t, PurchaseRequest> &itemsToBuy);
  ~BuyingItems() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"BuyingItems"};
  std::map<uint32_t, PurchaseRequest> itemsToBuy_;
  bool waitingOnBuyResponse_{false};
  bool waitingOnItemMovementResponse_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_BUYING_ITEMS_HPP_