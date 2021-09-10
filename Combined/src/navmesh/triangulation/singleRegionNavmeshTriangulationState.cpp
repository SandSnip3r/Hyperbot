#include "singleRegionNavmeshTriangulationState.hpp"

namespace navmesh {

namespace triangulation {

bool operator==(const ObjectData &a, const ObjectData &b) {
  return (a.objectAreaId == b.objectAreaId) && (a.objectInstanceId == b.objectInstanceId);
}

bool operator<(const ObjectData &a, const ObjectData &b) {
  if (a.objectInstanceId == b.objectInstanceId) {
    return a.objectAreaId < b.objectAreaId;
  } else {
    return a.objectInstanceId < b.objectInstanceId;
  }
}

} // namespace triangulation

} // namespace navmesh