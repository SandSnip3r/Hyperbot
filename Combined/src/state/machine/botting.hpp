#ifndef STATE_MACHINE_BOTTING_HPP_
#define STATE_MACHINE_BOTTING_HPP_

#include "stateMachine.hpp"

#include "entity/geometry.hpp"

#include <silkroad_lib/position.h>

#include <memory>

namespace state::machine {

class Botting : public StateMachine {
public:
  Botting(Bot &bot);
  ~Botting() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"Botting"};
  static const sro::Position kCenterOfJangan_;
  std::unique_ptr<StateMachine> childState_;
  sro::Position trainingSpotCenter_;
  std::unique_ptr<entity::Geometry> trainingAreaGeometry_;
  void initializeChildState();
  bool needToGoToTown() const;
};

} // namespace state::machine

#endif // STATE_MACHINE_BOTTING_HPP_