#ifndef STATE_MACHINE_BUYING_ITEMS_HPP_
#define STATE_MACHINE_BUYING_ITEMS_HPP_

#include "stateMachine.hpp"

#include <cstdint>
#include <vector>

#include <silkroad_lib/pk2/ref/item.hpp>

namespace state::machine {

class GmCommandSpawnAndPickItems : public StateMachine {
public:
  struct ItemRequest {
    sro::pk2::ref::ItemId refItemId;
    uint16_t quantity;
  };

  class ItemListBuilder {
  public:
    ItemListBuilder() = default;
    ItemListBuilder& addItemRequest(ItemRequest itemRequest) {
      itemRequests_.push_back(itemRequest);
      return *this;
    }
    std::vector<ItemRequest> getItemRequests() const { return itemRequests_; }
  private:
    std::vector<ItemRequest> itemRequests_;
  };

  GmCommandSpawnAndPickItems(Bot &bot, const std::vector<ItemRequest> &items);
  ~GmCommandSpawnAndPickItems() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"GmCommandSpawnAndPickItems"};
  // std::map<uint32_t, PurchaseRequest> itemsToBuy_;
  // bool waitingOnBuyResponse_{false};
  // bool waitingOnItemMovementResponse_{false};
  bool done_{false};
  std::vector<ItemRequest> items_;
  sro::Position originalPosition_;
  bool waitingForItemToSpawn_{false};
  size_t currentIndex_{0};

  void spawnNextItem();
};

} // namespace state::machine

#endif // STATE_MACHINE_BUYING_ITEMS_HPP_