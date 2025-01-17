#include "pk2/regionInfo.h"
#include "position_math.h"

namespace sro::pk2 {

RegionInfo::Region::RegionRect::RegionRect(Sector xSector, Sector zSector) : RegionRect(xSector, zSector, 0.0, 0.0, 1920.0, 1920.0) {
  //
}

RegionInfo::Region::RegionRect::RegionRect(Sector xSector, Sector zSector, float rectStartX, float rectStartZ, float rectEndX, float rectEndZ) :
      rectStart(position_math::worldRegionIdFromSectors(xSector, zSector), std::min(rectStartX, rectEndX), 0.0, std::min(rectStartZ, rectEndZ)),
      width(std::abs(rectEndX - rectStartX)),
      height(std::abs(rectEndZ - rectStartZ)) {
  //
}

} // namespace sro::pk2