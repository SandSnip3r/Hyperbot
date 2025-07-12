#ifndef RL_OBSERVATION_BUILDER_HPP_
#define RL_OBSERVATION_BUILDER_HPP_

#include <silkroad_lib/scalar_types.hpp>

// Foward declarations
class Bot;
namespace rl {
class Observation;

class ObservationBuilder {
public:
  static void buildObservationFromBot(const Bot &bot,
                              Observation &observation,
                              sro::scalar_types::EntityGlobalId opponentGlobalId);
};

} // namespace rl

#endif // RL_OBSERVATION_BUILDER_HPP_