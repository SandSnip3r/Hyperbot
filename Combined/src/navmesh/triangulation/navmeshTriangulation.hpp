#ifndef NAVMESH_TRIANGULATION_NAVMESH_TRIANGULATION_HPP_
#define NAVMESH_TRIANGULATION_NAVMESH_TRIANGULATION_HPP_

#include "math/vector.hpp"
#include "pk2/parsing/navmeshParser.hpp"
#include "singleRegionNavmeshTriangulation.hpp"

#include "vector.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace navmesh {

namespace triangulation {

namespace geometry_helpers {

bool lineTrimToRegion(math::Vector &p1, math::Vector &p2, const double minX, const double minY, const double maxX, const double maxY);

} // namespace geometry_helpers

class NavmeshTriangulation : public pathfinder::navmesh::NavmeshTypes<uint32_t> {
public:
  using State = SingleRegionNavmeshTriangulationState<IndexType>;
  using MarkerType = int;

  NavmeshTriangulation(const Navmesh &navmesh);
  void postProcess();
  void fixRegionObjectInstanceAssociations();
  const SingleRegionNavmeshTriangulation& getNavmeshTriangulationForRegion(const uint16_t regionId) const;

  using NavmeshTriangulationMapType = std::unordered_map<uint16_t, SingleRegionNavmeshTriangulation>;
  const NavmeshTriangulationMapType& getNavmeshTriangulationMap() const;

  void setOriginRegion(const uint16_t regionId);
  uint16_t getOriginRegion() const;
  math::Vector transformRegionPointIntoAbsolute(const math::Vector &point, const uint16_t regionId) const;
  std::pair<uint16_t, math::Vector> transformAbsolutePointIntoRegion(const math::Vector &point) const;

  // Pathfinder functions
  std::optional<IndexType> findTriangleForPoint(const pathfinder::Vector &point) const;
  State createStartState(const math::Vector &startPoint, const IndexType startTriangle) const;
  State createGoalState(const math::Vector &goalPoint, const IndexType goalTriangle) const;
  TriangleVertexIndicesType getTriangleVertexIndices(const IndexType triangleIndex) const;
  TriangleEdgeIndicesType getTriangleEdgeIndices(const IndexType triangleIndex) const;
  MarkerType getVertexMarker(const IndexType vertexIndex) const;
  pathfinder::Vector getVertex(const IndexType vertexIndex) const;
  TriangleVerticesType getTriangleVertices(const IndexType triangleIndex) const;
  std::vector<State> getSuccessors(const State &currentState, const State &goalState, const double agentRadius) const;
  EdgeType getSharedEdge(const IndexType triangle1Index, const IndexType triangle2Index) const;
  EdgeType getEdge(const IndexType edgeIndex) const;
  // Debug help (for Pathfinder)
  std::optional<IndexType> getVertexIndex(const pathfinder::Vector &vertex) const;
  
  static pathfinder::Vector to2dPoint(const math::Vector &point);
private:
  NavmeshTriangulationMapType navmeshTriangulationMap_;
  uint16_t originRegionId_{16512}; // Arbitrarily chose the center region of the entire map
  struct GlobalEdgeAndTriangleIndices {
    IndexType edgeIndex, triangleIndex;
  };
  std::unordered_map<IndexType, GlobalEdgeAndTriangleIndices> globalEdgeAndTriangleLinkMap_;
  std::optional<GlobalEdgeAndTriangleIndices> getNeighborTriangleAndEdge(const IndexType edgeIndex) const;
  
  void buildGlobalEdgesBasedOnBlockedTerrain(const Navmesh &navmesh, const Region &region, std::vector<navmesh::Edge> &globalEdges);
  void func(const Navmesh &navmesh, const Region &region);
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


} // namespace triangulation

} // namespace navmesh

#endif // NAVMESH_TRIANGULATION_NAVMESH_TRIANGULATION_HPP_