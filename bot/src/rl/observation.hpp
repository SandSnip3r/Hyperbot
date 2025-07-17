#ifndef RL_OBSERVATION_HPP_
#define RL_OBSERVATION_HPP_

#include "rl/items.hpp"
#include "rl/model_data/modelData.hpp"
#include "rl/skills.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

namespace rl {

// Forward declaration for friend declaration.
class ObservationBuilder;

class Observation {
public:
  std::string toString() const;

  // Returns the count of f32s in the observation.
  static consteval size_t size() {
    size_t sum = 0;
    sum += decltype(skillData_)::value_type::size() * std::tuple_size<decltype(skillData_)>();
    sum += decltype(itemData_)::value_type::size() * std::tuple_size<decltype(itemData_)>();
    sum += decltype(ourHpData_)::size();
    // sum += decltype(ourMpData_)::size();
    sum += decltype(opponentHpData_)::size();
    return sum;
  }

  // Writes the observation to a raw buffer of floats.
  // Returns the number of floats written to the buffer.
  size_t writeToArray(float *buffer) const;

  // Getters for reward calculation.
  uint32_t ourCurrentHp() const;
  uint32_t ourMaxHp() const;
  uint32_t opponentCurrentHp() const;
  uint32_t opponentMaxHp() const;

  // Public timestamp for other model input.
  std::chrono::steady_clock::time_point timestamp_;

private:
  friend class ObservationBuilder;

  std::array<model_data::SkillModelData, kSkillIdsForObservations.size()> skillData_;
  std::array<model_data::ItemModelData, kItemIdsForObservations.size()> itemData_;
  model_data::VitalModelData ourHpData_;
  // model_data::VitalModelData ourMpData_;
  model_data::VitalModelData opponentHpData_;
};

} // namespace rl

#endif // RL_OBSERVATION_HPP_