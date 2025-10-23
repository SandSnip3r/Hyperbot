#ifndef RL_SKILLS_HPP_
#define RL_SKILLS_HPP_

#include <silkroad_lib/scalar_types.hpp>

#include <array>

namespace rl {

constexpr std::array<sro::scalar_types::ReferenceSkillId, 0> kSkillIdsForObservations = {
  // sro::scalar_types::ReferenceSkillId{37}, // Snake Sword Dance (ranged, multi-target)
  // sro::scalar_types::ReferenceSkillId{300} // Stab Smash
  // sro::scalar_types::ReferenceSkillId{588} // Soul Cut Blade (ranged)
  // sro::scalar_types::ReferenceSkillId{1380} // Extreme Fire force
};

} // namespace rl

#endif // RL_SKILLS_HPP_