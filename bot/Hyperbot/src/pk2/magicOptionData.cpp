#include "magicOptionData.hpp"

namespace pk2 {

void MagicOptionData::addItem(ref::MagicOption &&magOpt) {
  magicOptions_.emplace(magOpt.id, magOpt);
}

const ref::MagicOption& MagicOptionData::getMagicOptionById(ref::MagicOptionId id) const {
  auto it = magicOptions_.find(id);
  if (it == magicOptions_.end()) {
    throw std::runtime_error("Trying to get magic option that does not exist");
  }
  return it->second;
}

} // namespace pk2