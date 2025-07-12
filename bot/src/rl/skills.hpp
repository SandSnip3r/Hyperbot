#ifndef RL_SKILLS_HPP_
#define RL_SKILLS_HPP_

#include <silkroad_lib/scalar_types.hpp>

#include <array>

namespace rl {

constexpr std::array kSkillIdsForObservations = {
  sro::scalar_types::ReferenceSkillId{300} // Stab Smash
  // sro::scalar_types::ReferenceSkillId{588} // Soul Cut Blade
};

} // namespace rl

#endif // RL_SKILLS_HPP_