#ifndef STATE_MACHINE_BOTTING_HPP_
#define STATE_MACHINE_BOTTING_HPP_

#include "stateMachine.hpp"

#include <silkroad_lib/position.h>

#include <memory>

namespace state::machine {

class Botting : public StateMachine {
public:
  Botting(Bot &bot);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
  void initialize();
  void reset();
private:
  std::unique_ptr<StateMachine> childState_;
  sro::Position trainingSpotCenter_;
};

} // namespace state::machine

#endif // STATE_MACHINE_BOTTING_HPP_