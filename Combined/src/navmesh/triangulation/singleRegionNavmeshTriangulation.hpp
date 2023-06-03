#ifndef NAVMESH_TRIANGULATION_SINGLE_REGION_NAVMESH_TRIANGULATION_HPP_
#define NAVMESH_TRIANGULATION_SINGLE_REGION_NAVMESH_TRIANGULATION_HPP_

#include "singleRegionNavmeshTriangulationState.hpp"
#include "navmesh/navmesh.hpp"

#include "triangle_lib_navmesh.h"
#include "vector.h"

#include <silkroad_lib/math/vector3.h>

#include <map>
#include <set>

namespace navmesh {

namespace triangulation {

enum class EdgeConstraintFlag : uint8_t {
  kNone = 0,
  kInternal = 1,
  kGlobal = 2,
  kBlocking = 4,
  kBridge = 8
};

struct ConstraintData {
  // Constraints are by default for the terrain unless constructed with object data
  ConstraintData() = default;
  ConstraintData(const ObjectData &objectData);
  std::optional<ObjectData> objectData_;
  EdgeConstraintFlag edgeFlag{EdgeConstraintFlag::kNone};
  std::optional<std::pair<uint32_t,bool>> linkIdAndIsSource_;
  bool is(const EdgeConstraintFlag flag) const;
  bool forTerrain() const;
  bool forObject() const;
  const ObjectData& getObjectData() const;
  bool hasLink() const;
  uint32_t getLinkId() const;
  bool isOnSourceSideOfLink() const;
};

std::ostream& operator<<(std::ostream &stream, const ConstraintData &data);
bool operator==(const ConstraintData &a, const ConstraintData &b);

struct ObjectLink {
  uint32_t srcObjectGlobalId, destObjectGlobalId;
  int16_t srcEdgeIndex, destEdgeIndex;
};

bool operator==(const ObjectLink &a, const ObjectLink &b);

EdgeConstraintFlag operator&(const EdgeConstraintFlag a, const EdgeConstraintFlag b);
EdgeConstraintFlag operator|(const EdgeConstraintFlag a, const EdgeConstraintFlag b);
EdgeConstraintFlag& operator|=(EdgeConstraintFlag &a, const EdgeConstraintFlag b);

class SingleRegionNavmeshTriangulation : public pathfinder::navmesh::TriangleLibNavmesh {
public:
  using State = SingleRegionNavmeshTriangulationState<IndexType>;

  SingleRegionNavmeshTriangulation(const navmesh::Navmesh &navmesh,
                                   const navmesh::Region &region,
                                   const triangle::triangleio &triangleData,
                                   const triangle::triangleio &triangleVoronoiData,
                                   std::vector<std::vector<ConstraintData>> &&constraintData,
                                   const std::vector<ObjectLink> &globalObjectLinks);
  std::vector<State> getNeighborsInObjectArea(const State &currentState) const;
  const std::vector<ConstraintData>& getEdgeConstraintData(const MarkerType edgeMarker) const;
  void setBlockedTerrainTriangles(std::vector<bool> &&blockedTriangles);
  bool terrainIsBlockedUnderTriangle(const IndexType triangleIndex) const;
  void addObjectDataForTriangle(const IndexType triangleIndex, const ObjectData &objectData);
  const std::vector<ObjectData>& getObjectDatasForTriangle(const IndexType triangleIndex) const;
  
  std::vector<State> getSuccessors(const State &currentState, const std::optional<State> goalState, const double agentRadius) const;
  bool agentFitsThroughEdge(const IndexType edgeIndex, const double agentRadius) const;
  static pathfinder::Vector to2dPoint(const sro::math::Vector3 &point);
  State createStartState(const sro::math::Vector3 &point, const IndexType triangleIndex) const;
  State createGoalState(const sro::math::Vector3 &point, const IndexType triangleIndex) const;
private:
  const navmesh::Navmesh &navmesh_;
  const navmesh::Region &region_;
  std::vector<std::vector<ConstraintData>> constraintData_;
  std::vector<ObjectLink> globalObjectLinks_;
  std::vector<bool> blockedTerrainTriangles_;
  std::vector<std::vector<ObjectData>> objectDatasForTriangles_;

  // Link data; exposed for visualization
public:
  using LinkIdType = decltype(std::declval<ConstraintData>().getLinkId());
  std::optional<LinkIdType> getLinkIdForTriangle(const IndexType triangleIndex) const;
private:
  struct LinkData {
    std::set<LinkIdType> accessibleTriangleIndices;
  };
  std::map<LinkIdType, LinkData> linkDataMap_;
  void buildLinkData();

  State createStateForPoint(const sro::math::Vector3 &point, const IndexType triangleIndex) const;
};
  
} // namespace pathfinder

} // namespace navmesh

#endif // NAVMESH_TRIANGULATION_SINGLE_REGION_NAVMESH_TRIANGULATION_HPP_
