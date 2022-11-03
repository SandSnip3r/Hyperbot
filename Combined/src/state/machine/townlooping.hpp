#ifndef STATE_MACHINE_TOWNLOOPING_HPP_
#define STATE_MACHINE_TOWNLOOPING_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/position.h>

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace state::machine {

class Townlooping : public StateMachine {
public:
  Townlooping(Bot &bot);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  std::map<uint32_t, int> shoppingList_;
  std::vector<Npc> npcsToVisit_;
  size_t currentNpcIndex_{0};
  std::unique_ptr<StateMachine> childState_;

  sro::Position positionOfNpc(Npc npc) const;
};

} // namespace state::machine

#endif // STATE_MACHINE_TOWNLOOPING_HPP_