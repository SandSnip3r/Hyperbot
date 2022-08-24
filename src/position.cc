#include "position.h"

namespace sro {

Position::Position() {}

Position::Position(RegionId regionId, float xOffset, float yOffset, float zOffset) {}

RegionId Position::regionId() const {
  return {};
}

Sector Position::xSector() const {
  return {};
}

Sector Position::zSector() const {
  return {};
}

float Position::xOffset() const {
  return {};
}

float Position::yOffset() const {
  return {};
}

float Position::zOffset() const {
  return {};
}

void Position::normalize() {
  
}

} // namespace sro