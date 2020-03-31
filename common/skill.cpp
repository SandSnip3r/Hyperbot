#include "skill.hpp"

namespace pk2::media {
  
bool Skill::isEfta() const {
  const int32_t kEfta = 1701213281;
  for (const auto param : params) {
    if (param == kEfta) {
      return true;
    }
  }
  return false;
}

} // namespace pk2::media
