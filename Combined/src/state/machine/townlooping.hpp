#ifndef STATE_MACHINE_TOWNLOOPING_HPP_
#define STATE_MACHINE_TOWNLOOPING_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace state::machine {

class Townlooping : public StateMachine {
public:
  Townlooping(Bot &bot);
  ~Townlooping() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"Townlooping"};
  std::map<uint32_t, int> shoppingList_;
  std::vector<Npc> npcsToVisit_;
  size_t currentNpcIndex_{0};
  std::vector<sro::scalar_types::ReferenceObjectId> buffsToUse_;

  bool sanityCheckUsedReturnScroll_{false};
  bool waitingForSpawn_{false};

  void buildBuffList();
  void buildShoppingList();
  void buildNpcList();
  std::optional<sro::scalar_types::ReferenceObjectId> getNextBuffToCast() const;
  sro::Position positionOfNpc(Npc npc) const;
};

} // namespace state::machine

#endif // STATE_MACHINE_TOWNLOOPING_HPP_