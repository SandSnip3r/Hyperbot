#include "teleportData.hpp"

#include <string>

namespace pk2::media {

void TeleportData::addTeleport(Teleport &&teleport) {
  teleports_.emplace(teleport.id, teleport);
}

bool TeleportData::haveTeleportWithId(TeleportId id) const {
  return (teleports_.find(id) != teleports_.end());
}

const Teleport& TeleportData::getTeleportById(TeleportId id) const {
  auto it = teleports_.find(id);
  if (it == teleports_.end()) {
    throw std::runtime_error("Trying to get non-existent teleport with id "+std::to_string(id));
  }
  return it->second;
}

const TeleportData::TeleportMap::size_type TeleportData::size() const {
  return teleports_.size();
}

} // namespace pk2::media