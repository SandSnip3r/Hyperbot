#ifndef STATE_MACHINE_TOWNLOOPING_HPP_
#define STATE_MACHINE_TOWNLOOPING_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/position.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace state::machine {

class Townlooping : public StateMachine {
public:
  Townlooping(Bot &bot);
  ~Townlooping() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"Townlooping"};

  enum class Town {
    kJangan,
    kConstantinople
  };

  bool soldItems_{false};
  std::vector<sro::scalar_types::StorageIndexType> slotsToSell_;
  std::map<uint32_t, int> shoppingList_;
  Town currentTown_;
  std::vector<Npc> npcsToVisit_;
  size_t currentNpcIndex_{0};
  std::vector<sro::scalar_types::ReferenceObjectId> buffsToUse_;

  bool sanityCheckUsedReturnScroll_{false};
  bool waitingForSpawn_{false};

  void buildBuffList();
  void buildShoppingList();
  void buildSellList();
  void buildNpcList();
  std::optional<sro::scalar_types::ReferenceObjectId> getNextBuffToCast() const;
  sro::Position positionOfNpc(Npc npc) const;
  bool done() const;
};

} // namespace state::machine

#endif // STATE_MACHINE_TOWNLOOPING_HPP_