#ifndef PK2_REGION_INFO_HPP_
#define PK2_REGION_INFO_HPP_

#include <silkroad_lib/position.h>

#include <vector>

namespace pk2 {

// These "regions" represent music zones. They're similar to safe zones, but not quite the same. They additionally include areas like fortresses.
struct RegionInfo {
  struct Region {
    enum class Type { kTown, kField };
    Type regionType;
    
    struct RegionRect {
      RegionRect(sro::Sector xSector, sro::Sector zSector);
      RegionRect(sro::Sector xSector, sro::Sector zSector, float rectStartX, float rectStartZ, float rectEndX, float rectEndZ);
      sro::Position rectStart;
      float width, height;
    };
    
    std::vector<RegionRect> regionRects;
  };
  
  std::vector<Region> regions;
};

} // namespace pk2

#endif // PK2_REGION_INFO_HPP_