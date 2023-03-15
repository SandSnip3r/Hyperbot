#include "refRegion.hpp"

#include <stdexcept>

namespace pk2 {

void RefRegion::addRegion(ref::Region &&region) {
  regions_.emplace(region.wRegionID, region);
}

const RefRegion::RegionMap& RefRegion::regionMap() const {
  return regions_;
}

const ref::Region& RefRegion::getRegion(sro::RegionId regionId) const {
  const auto it = regions_.find(regionId);
  if (it == regions_.end()) {
    throw std::runtime_error("Asking for region in RefRegion which does not exist");
  }
  return it->second;
}

} // namespace pk2