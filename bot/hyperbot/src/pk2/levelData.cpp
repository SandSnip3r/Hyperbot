#include "levelData.hpp"

#include <absl/strings/str_format.h>

namespace pk2 {

void LevelData::addLevelItem(sro::pk2::ref::Level &&level) {
  Levels_.emplace(level.lvl, level);
}

const sro::pk2::ref::Level& LevelData::getLevel(uint8_t lvl) const {
  auto it = Levels_.find(lvl);
  if (it == Levels_.end()) {
    throw std::runtime_error(absl::StrFormat("Trying to get level (%d) that does not exist", lvl));
  }
  return it->second;
}

} // namespace pk2