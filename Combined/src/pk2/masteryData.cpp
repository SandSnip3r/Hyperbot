#include "masteryData.hpp"
#include "logging.hpp"

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

} // namespace pk2