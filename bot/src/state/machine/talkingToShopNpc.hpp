#ifndef STATE_MACHINE_TALKING_TO_SHOP_NPC_HPP_
#define STATE_MACHINE_TALKING_TO_SHOP_NPC_HPP_

#include "buyingItems.hpp"
#include "stateMachine.hpp"

#include <cstdint>
#include <map>

namespace state::machine {

class TalkingToShopNpc : public StateMachine {
public:
  TalkingToShopNpc(Bot &bot, Npc npc, const std::map<uint32_t, int> &shoppingList, const std::vector<sro::scalar_types::StorageIndexType> &slotsToSell = {});
  ~TalkingToShopNpc() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"TalkingToShopNpc"};
  Npc npc_;
  const std::map<uint32_t, int> &shoppingList_;
  uint32_t npcGid_;
  std::map<uint32_t, BuyingItems::PurchaseRequest> itemsToBuy_;
  std::vector<sro::scalar_types::StorageIndexType> slotsToSell_;
  bool doneBuyingItems_{false};
  bool doneSellingItems_{false};
  bool waitingForSelectionResponse_{false};
  bool waitingForTalkResponse_{false};
  bool waitingForRepairResponse_{false};
  bool waitingOnStopTalkResponse_{false};
  bool waitingOnDeselectionResponse_{false};
  bool done_{false};

  void figureOutWhatToBuy();
  bool needToRepair() const;
  bool doneSellingItems() const;
  bool doneBuyingItems() const;
  bool doneWithNpc() const;
};

} // namespace state::machine

#endif // STATE_MACHINE_TALKING_TO_SHOP_NPC_HPP_