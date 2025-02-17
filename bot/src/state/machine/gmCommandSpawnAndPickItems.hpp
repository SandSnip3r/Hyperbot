#ifndef STATE_MACHINE_BUYING_ITEMS_HPP_
#define STATE_MACHINE_BUYING_ITEMS_HPP_

#include "bot.hpp"
#include "stateMachine.hpp"

#include <cstdint>
#include <vector>

#include <silkroad_lib/pk2/ref/item.hpp>

namespace state::machine {

class GmCommandSpawnAndPickItems : public StateMachine {
public:
  GmCommandSpawnAndPickItems(Bot &bot, const std::vector<Bot::ItemRequirement> &items);
  ~GmCommandSpawnAndPickItems() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"GmCommandSpawnAndPickItems"};
  bool initialized_{false};
  std::vector<Bot::ItemRequirement> items_;
  sro::Position originalPosition_;
  bool waitingForItemToSpawn_{false};

  Status spawnNextItem();
};

} // namespace state::machine

#endif // STATE_MACHINE_BUYING_ITEMS_HPP_