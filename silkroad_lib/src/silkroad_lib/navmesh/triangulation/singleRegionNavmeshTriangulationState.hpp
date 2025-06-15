#ifndef SRO_NAVMESH_TRIANGULATION_SINGLE_REGION_NAVMESH_TRIANGULATION_STATE_H_
#define SRO_NAVMESH_TRIANGULATION_SINGLE_REGION_NAVMESH_TRIANGULATION_STATE_H_

#include <pathfinder/navmesh_astar_state.h>

#include <cstdint>
#include <optional>
#include <ostream>

namespace sro::navmesh {

namespace triangulation {

struct ObjectData {
  uint32_t objectInstanceId;
  uint32_t objectAreaId;
};

template<typename IndexType>
class SingleRegionNavmeshTriangulationState : public pathfinder::navmesh::NavmeshAStarState<IndexType> {
public:
  using pathfinder::navmesh::NavmeshAStarState<IndexType>::NavmeshAStarState;
  void setObjectData(const ObjectData &objectData);
  bool isOnObject() const;
  void setOnTerrain();
  const ObjectData& getObjectData() const;

  bool isSameTriangleAs(const SingleRegionNavmeshTriangulationState<IndexType> &otherState) const;

  bool isTraversingLink() const;
  void resetLinkId();
  void setLinkId(const IndexType id);
  IndexType getLinkId() const;

  friend struct std::hash<SingleRegionNavmeshTriangulationState<IndexType>>;
private:
  std::optional<ObjectData> objectData_;
  std::optional<IndexType> linkId_;
};

bool operator==(const ObjectData &a, const ObjectData &b);
bool operator!=(const ObjectData &a, const ObjectData &b);
bool operator<(const ObjectData &a, const ObjectData &b);

#include "singleRegionNavmeshTriangulationState.inl"

} // namespace triangulation

} // namespace sro::navmesh

namespace std {

template<>
struct hash<sro::navmesh::triangulation::ObjectData> {
  std::size_t operator()(const sro::navmesh::triangulation::ObjectData &objData) const {
    return std::hash<uint32_t>()(objData.objectInstanceId) ^
           (std::hash<uint32_t>()(objData.objectAreaId) << 1);
  }
};

template <typename IndexType>
struct hash<sro::navmesh::triangulation::SingleRegionNavmeshTriangulationState<IndexType>> {
  std::size_t operator()(const sro::navmesh::triangulation::SingleRegionNavmeshTriangulationState<IndexType> &state) const {
    if (state.isGoal_) {
      return std::hash<bool>()(state.isGoal_);
    } else {
      return std::hash<pathfinder::navmesh::NavmeshAStarState<IndexType>>()(static_cast<pathfinder::navmesh::NavmeshAStarState<IndexType>>(state)) ^
             (std::hash<std::optional<sro::navmesh::triangulation::ObjectData>>()(state.objectData_) << 1);
    }
  }
};

} // namespace std

#endif // SRO_NAVMESH_TRIANGULATION_SINGLE_REGION_NAVMESH_TRIANGULATION_STATE_H_