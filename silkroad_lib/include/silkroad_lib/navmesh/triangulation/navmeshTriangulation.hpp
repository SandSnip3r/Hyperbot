#ifndef SRO_NAVMESH_TRIANGULATION_NAVMESH_TRIANGULATION_H_
#define SRO_NAVMESH_TRIANGULATION_NAVMESH_TRIANGULATION_H_

#include "math/vector3.hpp"
#include "pk2/navmeshParser.hpp"
#include "singleRegionNavmeshTriangulation.hpp"

#include "vector.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace sro::navmesh {

namespace triangulation {

namespace geometry_helpers {

bool lineTrimToRegion(math::Vector3 &p1, math::Vector3 &p2, const double minX, const double minY, const double maxX, const double maxY);

} // namespace geometry_helpers


class NavmeshTriangulation : public pathfinder::navmesh::NavmeshTypes<uint32_t> {
public:
  using State = SingleRegionNavmeshTriangulationState<IndexType>;
  using MarkerType = int;

  NavmeshTriangulation(const Navmesh &navmesh);
  const SingleRegionNavmeshTriangulation& getNavmeshTriangulationForRegion(const uint16_t regionId) const;

  using NavmeshTriangulationMapType = std::unordered_map<uint16_t, SingleRegionNavmeshTriangulation>;
  const NavmeshTriangulationMapType& getNavmeshTriangulationMap() const;

  void setOriginRegion(const uint16_t regionId);
  uint16_t getOriginRegion() const;
  math::Vector3 transformRegionPointIntoAbsolute(const math::Vector3 &point, const uint16_t regionId) const;
  std::pair<uint16_t, math::Vector3> transformAbsolutePointIntoRegion(const math::Vector3 &point) const;

  // Pathfinder functions
  std::optional<IndexType> findTriangleForPoint(const pathfinder::Vector &point) const;
  State createStartState(const math::Vector3 &startPoint, const IndexType startTriangle) const;
  State createGoalState(const math::Vector3 &goalPoint, const IndexType goalTriangle) const;
  TriangleVertexIndicesType getTriangleVertexIndices(const IndexType triangleIndex) const;
  TriangleEdgeIndicesType getTriangleEdgeIndices(const IndexType triangleIndex) const;
  MarkerType getVertexMarker(const IndexType vertexIndex) const;
  pathfinder::Vector getVertex(const IndexType vertexIndex) const;
  TriangleVerticesType getTriangleVertices(const IndexType triangleIndex) const;
  std::vector<State> getSuccessors(const State &currentState, const std::optional<State> goalState, const double agentRadius) const;
  std::vector<State> getNeighborsInObjectArea(const State &currentState) const;
  EdgeType getSharedEdge(const IndexType triangle1Index, const IndexType triangle2Index) const;
  EdgeType getEdge(const IndexType edgeIndex) const;
  EdgeVertexIndicesType getEdgeVertexIndices(const IndexType edgeIndex) const;
  // Debug help (for Pathfinder)
  std::optional<IndexType> getVertexIndex(const pathfinder::Vector &vertex) const;

  static pathfinder::Vector to2dPoint(const math::Vector3 &point);
private:
  NavmeshTriangulationMapType navmeshTriangulationMap_;
  uint16_t originRegionId_{16512}; // Arbitrarily chose the center region of the entire map
  std::vector<ObjectLink> linkData_;

  void postProcess(const Navmesh &navmesh);
  void linkGlobalEdgesBetweenRegions();
  void markObjectsAndAreasInCells(const Navmesh &navmesh);

  struct GlobalEdgeAndTriangleIndices {
    IndexType edgeIndex, triangleIndex;
  };
  std::unordered_map<IndexType, GlobalEdgeAndTriangleIndices> globalEdgeAndTriangleLinkMap_;
  std::optional<GlobalEdgeAndTriangleIndices> getNeighborTriangleAndEdge(const IndexType edgeIndex) const;

  void addObjectDataForTriangle(const IndexType triangleIndex, const ObjectData &objectData);
  void buildGlobalEdgesBasedOnBlockedTerrain(const Navmesh &navmesh, const Region &region, std::vector<navmesh::Edge> &globalEdges);
  void buildNavmeshForRegion(const Navmesh &navmesh, const Region &region);
  void markBlockedTerrainCells(SingleRegionNavmeshTriangulation &navmeshTriangulation, const Region &region) const;
  void markObjectsAndAreasInCells(SingleRegionNavmeshTriangulation &navmeshTriangulation, const Navmesh &navmesh, const Region &region) const;

  pathfinder::Vector translatePointToGlobal(pathfinder::Vector point, const uint16_t regionId) const;
  std::pair<uint16_t,pathfinder::Vector> translatePointToRegion(const pathfinder::Vector &point) const;
  bool regionExists(const uint16_t regionId) const;

  static std::pair<uint16_t, SingleRegionNavmeshTriangulation::IndexType> splitRegionAndIndex(const IndexType index);
  static IndexType createIndex(const SingleRegionNavmeshTriangulation::IndexType index, const uint16_t regionId);
  static SingleRegionNavmeshTriangulation::State createRegionState(const State &state);
  static State createGlobalState(const SingleRegionNavmeshTriangulation::State &regionState, const uint16_t regionId);
};

template<typename T>
class SortedPair {
public:
  SortedPair(const T &v_1, const T &v_2) : first(std::min(v_1,v_2)), second(std::max(v_1,v_2)) {}
  const T first, second;
};

template<typename T>
inline bool operator<(const SortedPair<T> &p1, const SortedPair<T> &p2) {
  if (p1.first == p2.first) {
    return p1.second < p2.second;
  } else {
    return p1.first < p2.first;
  }
}


} // namespace triangulation

} // namespace sro::navmesh

#endif // SRO_NAVMESH_TRIANGULATION_NAVMESH_TRIANGULATION_H_
