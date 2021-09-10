#ifndef NAVMESH_TRIANGULATION_SINGLE_REGION_NAVMESH_TRIANGULATION_HPP_
#define NAVMESH_TRIANGULATION_SINGLE_REGION_NAVMESH_TRIANGULATION_HPP_

#include "singleRegionNavmeshTriangulationState.hpp"
#include "math/vector.hpp"
#include "navmesh/navmesh.hpp"

#include "triangle_lib_navmesh.h"
#include "vector.h"

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
  bool is(const EdgeConstraintFlag flag) const;
  bool forTerrain() const;
  bool forObject() const;
  const ObjectData& getObjectData() const;
};

EdgeConstraintFlag operator&(const EdgeConstraintFlag a, const EdgeConstraintFlag b);
EdgeConstraintFlag operator|(const EdgeConstraintFlag a, const EdgeConstraintFlag b);
EdgeConstraintFlag& operator|=(EdgeConstraintFlag &a, const EdgeConstraintFlag b);

class SingleRegionNavmeshTriangulation : public pathfinder::navmesh::TriangleLibNavmesh {
public:
  using State = SingleRegionNavmeshTriangulationState<IndexType>;

  SingleRegionNavmeshTriangulation(const navmesh::Navmesh &navmesh, const navmesh::Region &region, const triangle::triangleio &triangleData, const triangle::triangleio &triangleVoronoiData, std::vector<ConstraintData> &&constraintData);
  std::vector<State> getNeighborsInObjectArea(const State &currentState) const;
  const ConstraintData& getEdgeConstraintData(const MarkerType edgeMarker) const;
  void setBlockedTerrainTriangles(std::vector<bool> &&blockedTriangles);
  bool terrainIsBlockedUnderTriangle(const IndexType triangleIndex) const;
  void addObjectDataForTriangle(const IndexType triangleIndex, const ObjectData &objectData);
  const std::vector<ObjectData>& getObjectDatasForTriangle(const IndexType triangleIndex) const;
  
  std::vector<State> getSuccessors(const State &currentState, const std::optional<State> goalState, const double agentRadius) const;
  bool agentFitsThroughEdge(const IndexType edgeIndex, const double agentRadius) const;
  static pathfinder::Vector to2dPoint(const math::Vector &point);
  State createStartState(const math::Vector &point, const IndexType triangleIndex) const;
  State createGoalState(const math::Vector &point, const IndexType triangleIndex) const;
private:
  const navmesh::Navmesh &navmesh_;
  const navmesh::Region &region_;
  std::vector<ConstraintData> constraintData_;
  std::vector<bool> blockedTerrainTriangles_;
  std::vector<std::vector<ObjectData>> objectDatasForTriangles_;

  State createStateForPoint(const math::Vector &point, const IndexType triangleIndex) const;
};
  
} // namespace pathfinder

} // namespace navmesh

#endif // NAVMESH_TRIANGULATION_SINGLE_REGION_NAVMESH_TRIANGULATION_HPP_
