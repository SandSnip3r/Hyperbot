#ifndef STATE_MACHINE_BOTTING_HPP_
#define STATE_MACHINE_BOTTING_HPP_

#include "stateMachine.hpp"

#include "entity/geometry.hpp"

#include <silkroad_lib/position.hpp>

namespace state::machine {

class Botting : public StateMachine {
public:
  Botting(Bot &bot);
  ~Botting() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"Botting"};
  sro::Position trainingSpotCenter_;
  double trainingRadius_;
  std::unique_ptr<entity::Geometry> trainingAreaGeometry_;
  void setTrainingSpotFromConfig();
  void initializeChildState();
};

} // namespace state::machine

#endif // STATE_MACHINE_BOTTING_HPP_