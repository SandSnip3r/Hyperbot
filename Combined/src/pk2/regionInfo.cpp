#include "regionInfo.hpp"

#include <silkroad_lib/position_math.h>

#include <iostream>

namespace pk2 {

RegionInfo::Region::RegionRect::RegionRect(sro::Sector xSector, sro::Sector zSector) : RegionRect(xSector, zSector, 0.0, 0.0, 1920.0, 1920.0) {
  //
}

RegionInfo::Region::RegionRect::RegionRect(sro::Sector xSector, sro::Sector zSector, float rectStartX, float rectStartZ, float rectEndX, float rectEndZ) :
      rectStart(sro::position_math::worldRegionIdFromSectors(xSector, zSector), std::min(rectStartX, rectEndX), 0.0, std::min(rectStartZ, rectEndZ)),
      width(std::abs(rectEndX - rectStartX)),
      height(std::abs(rectEndZ - rectStartZ)) {
  //
}

} // namespace pk2