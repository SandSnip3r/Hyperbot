#include "playerCharacter.hpp"

#include <absl/strings/str_format.h>

namespace entity {

std::string PlayerCharacter::toStringImpl(const sro::pk2::GameData *gameData) const {
  return absl::StrFormat("%s", name);
}

} // namespace entity