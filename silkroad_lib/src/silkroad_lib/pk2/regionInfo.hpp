#ifndef SRO_PK2_REGION_INFO_HPP_
#define SRO_PK2_REGION_INFO_HPP_

#include <silkroad_lib/position.hpp>

#include <vector>

namespace sro::pk2 {

// These "regions" represent music zones. They're similar to safe zones, but not quite the same. They additionally include areas like fortresses.
struct RegionInfo {
  struct Region {
    enum class Type { kTown, kField };
    Type regionType;

    struct RegionRect {
      RegionRect(Sector xSector, Sector zSector);
      RegionRect(Sector xSector, Sector zSector, float rectStartX, float rectStartZ, float rectEndX, float rectEndZ);
      Position rectStart;
      float width, height;
    };

    std::vector<RegionRect> regionRects;
  };

  std::vector<Region> regions;
};

} // namespace sro::pk2

#endif // SRO_PK2_REGION_INFO_HPP_