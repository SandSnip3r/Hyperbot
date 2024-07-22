#include "masteryData.hpp"

#include <absl/strings/str_format.h>

#include <stdexcept>
#include <string>

namespace pk2 {

void MasteryData::addMastery(pk2::ref::Mastery &&mastery) {
  masteries_.emplace(mastery.masteryId, mastery);
}

const pk2::ref::Mastery& MasteryData::getMasteryById(pk2::ref::MasteryId id) const {
  auto it = masteries_.find(id);
  if (it == masteries_.end()) {
    throw std::runtime_error("Trying to get non-existent mastery with id "+std::to_string(id));
  }
  return it->second;
}

pk2::ref::MasteryId MasteryData::getMasteryIdByMasteryNameCode(std::string_view masteryNameCode) const {
  for (const auto &idMasteryPair : masteries_) {
    if (idMasteryPair.second.masteryNameCode == masteryNameCode) {
      return idMasteryPair.first;
    }
  }
  throw std::runtime_error(absl::StrFormat("Cannot find mastery id for MasteryNameCode \"%s\"", masteryNameCode));
}

} // namespace pk2