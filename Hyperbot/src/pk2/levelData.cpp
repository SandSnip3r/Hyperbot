#include "levelData.hpp"

namespace pk2 {

void LevelData::addLevelItem(ref::Level &&level) {
  Levels_.emplace(level.lvl, level);
}

const ref::Level& LevelData::getLevel(uint8_t lvl) const {
  auto it = Levels_.find(lvl);
  if (it == Levels_.end()) {
    throw std::runtime_error("Trying to get level that does not exist");
  }
  return it->second;
}

} // namespace pk2