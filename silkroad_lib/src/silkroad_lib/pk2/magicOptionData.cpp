#include "magicOptionData.hpp"

namespace sro::pk2 {

void MagicOptionData::addItem(sro::pk2::ref::MagicOption &&magOpt) {
  magicOptions_.emplace(magOpt.id, magOpt);
}

const sro::pk2::ref::MagicOption& MagicOptionData::getMagicOptionById(sro::pk2::ref::MagicOptionId id) const {
  auto it = magicOptions_.find(id);
  if (it == magicOptions_.end()) {
    throw std::runtime_error("Trying to get magic option that does not exist");
  }
  return it->second;
}

} // namespace sro::pk2
