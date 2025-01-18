#include "teleportData.hpp"

#include <stdexcept>
#include <string>

namespace pk2 {

void TeleportData::addTeleport(sro::pk2::ref::Teleport &&teleport) {
  teleports_.emplace(teleport.id, teleport);
}

bool TeleportData::haveTeleportWithId(sro::pk2::ref::TeleportId id) const {
  return (teleports_.find(id) != teleports_.end());
}

const sro::pk2::ref::Teleport& TeleportData::getTeleportById(sro::pk2::ref::TeleportId id) const {
  auto it = teleports_.find(id);
  if (it == teleports_.end()) {
    throw std::runtime_error("Trying to get non-existent teleport with id "+std::to_string(id));
  }
  return it->second;
}

const TeleportData::TeleportMap::size_type TeleportData::size() const {
  return teleports_.size();
}

} // namespace pk2