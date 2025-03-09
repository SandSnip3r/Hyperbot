#ifndef STATE_MACHINE_BUYING_ITEMS_HPP_
#define STATE_MACHINE_BUYING_ITEMS_HPP_

#include "bot.hpp"
#include "broker/eventBroker.hpp"
#include "common/itemRequirement.hpp"
#include "stateMachine.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <silkroad_lib/pk2/ref/item.hpp>

namespace state::machine {

class GmCommandSpawnAndPickItems : public StateMachine {
public:
  // TODO: Create a move constructor for the items.
  GmCommandSpawnAndPickItems(StateMachine *parent, const std::vector<common::ItemRequirement> &items);
  ~GmCommandSpawnAndPickItems() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"GmCommandSpawnAndPickItems"};
  bool initialized_{false};
  std::vector<common::ItemRequirement> items_;
  sro::Position originalPosition_;
  std::optional<broker::EventBroker::EventId> requestTimeoutEventId_;

  Status spawnNextItem();
};

} // namespace state::machine

#endif // STATE_MACHINE_BUYING_ITEMS_HPP_