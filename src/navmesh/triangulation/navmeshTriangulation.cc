#include "math/matrix4x4.h"
#include "navmesh/triangulation/navmeshTriangulation.h"
#include "position_math.h"

#include "behaviorBuilder.h"
#include "math_helpers.h"
#include "triangle/triangle_api.h"

#include <fstream>
#include <iostream>
#include <set>
#include <stack>
#include <unordered_set>

namespace sro::navmesh::triangulation {

NavmeshTriangulation::NavmeshTriangulation(const Navmesh &navmesh) {
  static_assert((std::is_same_v<SingleRegionNavmeshTriangulation::IndexType, uint16_t>), "SingleRegionNavmeshTriangulation::IndexType must be of type uint16_t (it is not)");
  // Triangulate every region in the navmesh
  const auto &regionMap = navmesh.getRegionMap();
  for (const auto &regionIdRegionPair : regionMap) {
    try {
      buildNavmeshForRegion(navmesh, regionIdRegionPair.second);
    } catch (...) {
      std::cout << "No" << std::endl;
    }
  }
  postProcess(navmesh);
}

void NavmeshTriangulation::postProcess(const Navmesh &navmesh) {
  linkGlobalEdgesBetweenRegions();
  markObjectsAndAreasInCells(navmesh);
}

void NavmeshTriangulation::linkGlobalEdgesBetweenRegions() {
  enum class Direction {
    kLeft,
    kRight,
    kTop,
    kBottom
  };
  auto getDirectionString = [](const Direction dir) -> std::string {
    if (dir == Direction::kLeft) {
      return "left";
    } else if (dir == Direction::kRight) {
      return "right";
    } else if (dir == Direction::kTop) {
      return "top";
    } else if (dir == Direction::kBottom) {
      return "bottom";
    }
  };
  auto getEdgeSide = [](const auto edge) {
    if (edge.first.x() == edge.second.x()) {
      // Edge is vertical
      if (edge.first.x() == 0.0) {
        return Direction::kLeft;
      } else if (edge.first.x() == 1920.0) {
        return Direction::kRight;
      } else {
        throw std::runtime_error("Vertical global edge is neither on the left nor right of the region");
      }
    } else if (edge.first.y() == edge.second.y()) {
      // Edge is horizontal
      if (edge.first.y() == 0.0) {
        return Direction::kBottom;
      } else if (edge.first.y() == 1920.0) {
        return Direction::kTop;
      } else {
        throw std::runtime_error("Horizontal global edge is neither on the left nor right of the region");
      }
    } else {
      throw std::runtime_error("Global edge is neither vertical nor horizontal");
    }
  };

  struct GlobalEdges {
    std::vector<SingleRegionNavmeshTriangulation::IndexType> leftGlobalEdges, rightGlobalEdges, topGlobalEdges, bottomGlobalEdges;
  };

  std::unordered_map<uint16_t, GlobalEdges> regionGlobalEdgesMap;

  for (const auto &regionIdTriangulationPair : navmeshTriangulationMap_) {
    const auto &navmeshTriangulation = regionIdTriangulationPair.second;
    auto &globalEdges = regionGlobalEdgesMap[regionIdTriangulationPair.first];
    for (SingleRegionNavmeshTriangulation::IndexType index=0; index<navmeshTriangulation.getEdgeCount(); ++index) {
      const auto edgeMarker = navmeshTriangulation.getEdgeMarker(index);
      if (edgeMarker >= 2) {
        const auto &edgeConstraintData = navmeshTriangulation.getEdgeConstraintData(edgeMarker);
        for (const auto &constraint : edgeConstraintData) {
          if (constraint.forTerrain() && constraint.is(EdgeConstraintFlag::kGlobal)) {
            // This is a global edge of the region
            const auto edge = navmeshTriangulation.getEdge(index);
            const auto edgeSide = getEdgeSide(edge);
            if (edgeSide == Direction::kLeft) {
              globalEdges.leftGlobalEdges.push_back(index);
            } else if (edgeSide == Direction::kRight) {
              globalEdges.rightGlobalEdges.push_back(index);
            } else if (edgeSide == Direction::kTop) {
              globalEdges.topGlobalEdges.push_back(index);
            } else if (edgeSide == Direction::kBottom) {
              globalEdges.bottomGlobalEdges.push_back(index);
            } else {
              throw std::runtime_error("Impossible edge direction");
            }
          }
        }
      }
    }
    std::sort(globalEdges.leftGlobalEdges.begin(), globalEdges.leftGlobalEdges.end(), [&navmeshTriangulation](const auto e1, const auto e2) {
      const auto edge1 = navmeshTriangulation.getEdge(e1);
      const auto edge2 = navmeshTriangulation.getEdge(e2);
      return std::min(edge1.first.y(), edge1.second.y()) < std::min(edge2.first.y(), edge2.second.y());
    });
    std::sort(globalEdges.rightGlobalEdges.begin(), globalEdges.rightGlobalEdges.end(), [&navmeshTriangulation](const auto e1, const auto e2) {
      const auto edge1 = navmeshTriangulation.getEdge(e1);
      const auto edge2 = navmeshTriangulation.getEdge(e2);
      return std::min(edge1.first.y(), edge1.second.y()) < std::min(edge2.first.y(), edge2.second.y());
    });
    std::sort(globalEdges.topGlobalEdges.begin(), globalEdges.topGlobalEdges.end(), [&navmeshTriangulation](const auto e1, const auto e2) {
      const auto edge1 = navmeshTriangulation.getEdge(e1);
      const auto edge2 = navmeshTriangulation.getEdge(e2);
      return std::min(edge1.first.x(), edge1.second.x()) < std::min(edge2.first.x(), edge2.second.x());
    });
    std::sort(globalEdges.bottomGlobalEdges.begin(), globalEdges.bottomGlobalEdges.end(), [&navmeshTriangulation](const auto e1, const auto e2) {
      const auto edge1 = navmeshTriangulation.getEdge(e1);
      const auto edge2 = navmeshTriangulation.getEdge(e2);
      return std::min(edge1.first.x(), edge1.second.x()) < std::min(edge2.first.x(), edge2.second.x());
    });
  }

  // Make sure that all of the neighboring regions have global edges that pair up
  struct NeighboringRegion {
    uint16_t id;
    Direction direction;
  };
  auto getNeighboringRegionIds = [](const uint16_t regionId) -> std::array<NeighboringRegion, 4> {
    const auto [regionX, regionY] = sro::position_math::sectorsFromWorldRegionId(regionId);
    return {{
      {sro::position_math::worldRegionIdFromSectors(regionX+1,regionY), Direction::kRight},
      {sro::position_math::worldRegionIdFromSectors(regionX-1,regionY), Direction::kLeft},
      {sro::position_math::worldRegionIdFromSectors(regionX,regionY+1), Direction::kTop},
      {sro::position_math::worldRegionIdFromSectors(regionX,regionY-1), Direction::kBottom}
    }};
  };
  auto edgesAllMatch = [&](const uint16_t thisRegionId, const auto &thisRegionEdges, const Direction direction, const uint16_t otherRegionId, const auto &otherRegionEdges) {
    const auto thisRegionTriangulationIt = navmeshTriangulationMap_.find(thisRegionId);
    const auto otherRegionTriangulationIt = navmeshTriangulationMap_.find(otherRegionId);
    if (thisRegionTriangulationIt == navmeshTriangulationMap_.end()) {
      throw std::runtime_error("Comparing regions, but missing triangulation");
    }
    if (otherRegionTriangulationIt == navmeshTriangulationMap_.end()) {
      throw std::runtime_error("Comparing regions, but missing triangulation");
    }
    const auto &thisRegionTriangulation = thisRegionTriangulationIt->second;
    const auto &otherRegionTriangulation = otherRegionTriangulationIt->second;
    auto getVal = [&direction](const auto point){
      if (direction == Direction::kLeft || direction == Direction::kRight) {
        return point.y();
      } else {
        return point.x();
      }
    };
    if (thisRegionEdges.size() != otherRegionEdges.size()) {
      // These two regions have different numbers of edges on their shared boundary
      return false;
    }
    for (std::size_t i=0; i<thisRegionEdges.size(); ++i) {
      // These two edges should be the same (but on opposite sides of their region)
      const auto thisRegionEdge = thisRegionTriangulation.getEdge(thisRegionEdges[i]);
      const auto otherRegionEdge = otherRegionTriangulation.getEdge(otherRegionEdges[i]);
      bool checkY;
      if (direction == Direction::kLeft) {
        if (thisRegionEdge.first.x() != 0.0 ||
            thisRegionEdge.second.x() != 0.0 ||
            otherRegionEdge.first.x() != 1920.0 ||
            otherRegionEdge.second.x() != 1920.0) {
          // These edges arent on the correct sides
          std::cout << "These edges arent on the correct sides (checking kLeft)" << std::endl;
          return false;
        }
        checkY = true;
      } else if (direction == Direction::kRight) {
        if (thisRegionEdge.first.x() != 1920.0 ||
            thisRegionEdge.second.x() != 1920.0 ||
            otherRegionEdge.first.x() != 0.0 ||
            otherRegionEdge.second.x() != 0.0) {
          // These edges arent on the correct sides
          std::cout << "These edges arent on the correct sides (checking kRight)" << std::endl;
          return false;
        }
        checkY = true;
      } else if (direction == Direction::kTop) {
        if (thisRegionEdge.first.y() != 1920.0 ||
            thisRegionEdge.second.y() != 1920.0 ||
            otherRegionEdge.first.y() != 0.0 ||
            otherRegionEdge.second.y() != 0.0) {
          // These edges arent on the correct sides
          std::cout << "These edges arent on the correct sides (checking kTop)" << std::endl;
          return false;
        }
        checkY = false;
      } else if (direction == Direction::kBottom) {
        if (thisRegionEdge.first.y() != 0.0 ||
            thisRegionEdge.second.y() != 0.0 ||
            otherRegionEdge.first.y() != 1920.0 ||
            otherRegionEdge.second.y() != 1920.0) {
          // These edges arent on the correct sides
          std::cout << "These edges arent on the correct sides (checking kBottom)" << std::endl;
          return false;
        }
        checkY = false;
      } else {
        throw std::runtime_error("Impossible direction");
      }
      auto haveSameValue = [checkY](const auto &e1p1, const auto &e1p2, const auto &e2p1, const auto &e2p2){
        // Found three bad cases: 0.001953, 0.020493, and 0.004517 (all of which are on the south wall of the jangan fortress)
        const double kPrecisionTolerance{0.0205};
        if (checkY) {
          return (pathfinder::math::equal(e1p1.y(), e2p1.y(), kPrecisionTolerance) && pathfinder::math::equal(e1p2.y(), e2p2.y(), kPrecisionTolerance));
        } else {
          return (pathfinder::math::equal(e1p1.x(), e2p1.x(), kPrecisionTolerance) && pathfinder::math::equal(e1p2.x(), e2p2.x(), kPrecisionTolerance));
        }
      };
      if (!(haveSameValue(thisRegionEdge.first, thisRegionEdge.second, otherRegionEdge.first, otherRegionEdge.second) ||
            haveSameValue(thisRegionEdge.first, thisRegionEdge.second, otherRegionEdge.second, otherRegionEdge.first))) {
        // Edges do not match
        return false;
      }
    }
    // Everything matched up
    return true;
  };
  auto getIndexOfTriangleConnectedToEdge = [&](const SingleRegionNavmeshTriangulation::IndexType edgeIndex, const uint16_t regionId) {
    const auto triangulationIt = navmeshTriangulationMap_.find(regionId);
    if (triangulationIt == navmeshTriangulationMap_.end()) {
      throw std::runtime_error("Getting index of triangle for edge, but region triangulation is missing");
    }
    const auto &regionTriangulation = triangulationIt->second;
    std::optional<SingleRegionNavmeshTriangulation::IndexType> result;
    const auto triangleCount = regionTriangulation.getTriangleCount();
    for (std::size_t triangleIndex=0; triangleIndex<triangleCount; ++triangleIndex) {
      const auto triangleEdgeIndices = regionTriangulation.getTriangleEdgeIndices(triangleIndex);
      if (std::get<0>(triangleEdgeIndices) == edgeIndex ||
          std::get<1>(triangleEdgeIndices) == edgeIndex ||
          std::get<2>(triangleEdgeIndices) == edgeIndex) {
        // This triangle has this edge (and it should be the only such triangle)
        if (result.has_value()) {
          // Already found a triangle for this edge, weird
          throw std::runtime_error("Multiple triangles exist for this edge, shouldnt be possible");
        }
        result = triangleIndex;
      }
    }
    if (!result.has_value()) {
      throw std::runtime_error("Could not find a triangle which has this edge");
    }
    return *result;
  };
  std::set<SortedPair<uint16_t>> alreadyCheckedSet;
  for (const auto &regionIdGlobalEdgesPair : regionGlobalEdgesMap) {
    const auto regionId = regionIdGlobalEdgesPair.first;
    const auto &globalEdges = regionIdGlobalEdgesPair.second;
    for (const auto neighboringRegion : getNeighboringRegionIds(regionId)) {
      if (alreadyCheckedSet.find({regionId, neighboringRegion.id}) == alreadyCheckedSet.end()) {
        auto it = regionGlobalEdgesMap.find(neighboringRegion.id);
        if (it != regionGlobalEdgesMap.end()) {
          // Neighboring region exists
          // Validate that the edges on the shared boundary match up
          const std::vector<SingleRegionNavmeshTriangulation::IndexType> *thisRegionEdges, *otherRegionEdges;
          if (neighboringRegion.direction == Direction::kLeft) {
            thisRegionEdges = &globalEdges.leftGlobalEdges;
            otherRegionEdges = &(it->second.rightGlobalEdges);
          } else if (neighboringRegion.direction == Direction::kRight) {
            thisRegionEdges = &globalEdges.rightGlobalEdges;
            otherRegionEdges = &(it->second.leftGlobalEdges);
          } else if (neighboringRegion.direction == Direction::kTop) {
            thisRegionEdges = &globalEdges.topGlobalEdges;
            otherRegionEdges = &(it->second.bottomGlobalEdges);
          } else if (neighboringRegion.direction == Direction::kBottom) {
            thisRegionEdges = &globalEdges.bottomGlobalEdges;
            otherRegionEdges = &(it->second.topGlobalEdges);
          } else {
            throw std::runtime_error("Impossible direction");
          }
          if (!edgesAllMatch(regionId, *thisRegionEdges, neighboringRegion.direction, neighboringRegion.id, *otherRegionEdges)) {
            throw std::runtime_error("Global edges of neighboring regions ("+std::to_string(regionId)+"&"+std::to_string(neighboringRegion.id)+") do not match");
          }
          // Now, build a mapping between each of the pairs of edges on the boundary
          // Find out which triangles are connected to these edges
          for (std::size_t i=0; i<thisRegionEdges->size(); ++i) {
            const auto triangleIndexForThisEdge = getIndexOfTriangleConnectedToEdge((*thisRegionEdges)[i], regionId);
            const auto triangleIndexForOtherEdge = getIndexOfTriangleConnectedToEdge((*otherRegionEdges)[i], neighboringRegion.id);
            const auto globalTriangleIndexForThisEdge = createIndex(triangleIndexForThisEdge, regionId);
            const auto globalTriangleIndexForOtherEdge = createIndex(triangleIndexForOtherEdge, neighboringRegion.id);
            const auto thisEdgeGlobalIndex = createIndex((*thisRegionEdges)[i], regionId);
            const auto otherEdgeGlobalIndex = createIndex((*otherRegionEdges)[i], neighboringRegion.id);
            globalEdgeAndTriangleLinkMap_.emplace(thisEdgeGlobalIndex, GlobalEdgeAndTriangleIndices{otherEdgeGlobalIndex, globalTriangleIndexForOtherEdge});
            globalEdgeAndTriangleLinkMap_.emplace(otherEdgeGlobalIndex, GlobalEdgeAndTriangleIndices{thisEdgeGlobalIndex, globalTriangleIndexForThisEdge});
          }
        }
        alreadyCheckedSet.insert({regionId, neighboringRegion.id});
      }
    }
  }
}

void NavmeshTriangulation::markObjectsAndAreasInCells(const Navmesh &navmesh) {
  std::set<uint32_t> alreadyMarkedObjectInstances;
  for (const auto &regionIdRegionPair : navmesh.getRegionMap()) {
    const auto regionId = regionIdRegionPair.first;
    const auto &region = regionIdRegionPair.second;
    for (const auto objectInstanceId : region.objectInstanceIds) {
      if (alreadyMarkedObjectInstances.find(objectInstanceId) != alreadyMarkedObjectInstances.end()) {
        // Already marked cells for this object
        continue;
      } else {
        alreadyMarkedObjectInstances.insert(objectInstanceId);
      }
      const auto &objectInstance = navmesh.getObjectInstance(objectInstanceId);
      const auto &objectResource = navmesh.getObjectResource(objectInstance.objectId);
      const auto transformationFromObjectFramToWorld = navmesh.getTransformationFromObjectInstanceToWorld(objectInstanceId, regionId);
      for (const auto &objectCell : objectResource.cells) {
        objectResource.vertices.at(objectCell.vertex0);
      }
      auto tmpAreaIds = objectResource.cellAreaIds;
      std::sort(tmpAreaIds.begin(), tmpAreaIds.end());
      auto newEnd = std::unique(tmpAreaIds.begin(), tmpAreaIds.end());
      for (auto areaIdIt = tmpAreaIds.begin(); areaIdIt != newEnd; ++areaIdIt) {
        // For each area in the object
        // Pick a point that is inside the instance, inside the area, AND within the region
        const std::optional<math::Vector3> point = [this, &regionId, &transformationFromObjectFramToWorld, &objectResource, &areaIdIt]() -> std::optional<math::Vector3> {
          // For each cell in the object, check if the center point is in the navmesh
          for (std::size_t cellIndex=0; cellIndex<objectResource.cells.size(); ++cellIndex) {
            if (objectResource.cells[cellIndex].eventZoneData.has_value()) {
              // TODO: This is an event zone; for now, ignore
              continue;
            }
            if (objectResource.cellAreaIds[cellIndex] == *areaIdIt) {
              // This cell is for our area
              const auto &cell = objectResource.cells[cellIndex];
              const auto centerPoint = [&objectResource, &cell]() -> math::Vector3 {
                const double avgX = (objectResource.vertices[cell.vertex0].x + objectResource.vertices[cell.vertex1].x + objectResource.vertices[cell.vertex2].x) / 3.0;
                const double avgZ = (objectResource.vertices[cell.vertex0].z + objectResource.vertices[cell.vertex1].z + objectResource.vertices[cell.vertex2].z) / 3.0;
                return {static_cast<float>(avgX), 0, static_cast<float>(avgZ)};
              }();
              // Check if this point is inside of an enabled region
              const auto absoluteCenterPoint = transformRegionPointIntoAbsolute(transformationFromObjectFramToWorld*centerPoint, regionId);
              const auto regionIdPointPair = transformAbsolutePointIntoRegion(absoluteCenterPoint);
              if (regionExists(regionIdPointPair.first)) {
                return centerPoint;
              }
            }
          }
          // No valid cell found
          return {};
        }();
        if (!point.has_value()) {
          continue;
        }

        const auto transformedPoint = transformationFromObjectFramToWorld*(*point);
        const auto globalPoint = transformRegionPointIntoAbsolute(transformedPoint, regionId);
        const auto triangleIndex = findTriangleForPoint({globalPoint.x, globalPoint.z});
        if (!triangleIndex.has_value()) {
          continue;
        }

        // BFS outward from this triangle and mark all triangles inside the object
        State startState(*triangleIndex);
        startState.setObjectData({objectInstanceId, *areaIdIt});
        std::stack<State> nextStates;
        std::set<IndexType> visitedTriangles;
        nextStates.push(startState);

        while (!nextStates.empty()) {
          const auto currentState = nextStates.top();
          nextStates.pop();
          if (visitedTriangles.find(currentState.getTriangleIndex()) != visitedTriangles.end()) {
            // Triangle already visited, skip
            continue;
          }

          // Mark this current triangle as "visited"
          visitedTriangles.emplace(currentState.getTriangleIndex());

          if (currentState.isOnObject() && currentState.getObjectData().objectInstanceId == objectInstanceId && currentState.getObjectData().objectAreaId == *areaIdIt) {
            // This state is for our current object and the area that we're in
            // Mark this triangle as being associated with a specific object
            addObjectDataForTriangle(currentState.getTriangleIndex(), currentState.getObjectData());
          } else {
            continue;
          }
          // Get the neighboring states from here
          const auto successors = getNeighborsInObjectArea(currentState);
          for (const auto &successorState : successors) {
            if (visitedTriangles.find(successorState.getTriangleIndex()) == visitedTriangles.end()) {
              // This successor triangle has not yet been visited
              nextStates.push(successorState);
            }
          }
        }
      }
    }
  }
}

void NavmeshTriangulation::addObjectDataForTriangle(const IndexType triangleIndex, const ObjectData &objectData) {
  const auto [regionId, index] = splitRegionAndIndex(triangleIndex);
  auto it = navmeshTriangulationMap_.find(regionId);
  if (it == navmeshTriangulationMap_.end()) {
    throw std::runtime_error("Trying to add object data for triangle which is a non-existent region");
  }
  it->second.addObjectDataForTriangle(index, objectData);
}

const SingleRegionNavmeshTriangulation& NavmeshTriangulation::getNavmeshTriangulationForRegion(const uint16_t regionId) const {
  auto it = navmeshTriangulationMap_.find(regionId);
  if (it == navmeshTriangulationMap_.end()) {
    throw std::runtime_error("Asking for navmesh triangulation for a non existent region");
  }
  return it->second;
}

void NavmeshTriangulation::setOriginRegion(const uint16_t regionId) {
  // TODO: If someone changes this and uses old global coordinates, there will be problems
  originRegionId_ = regionId;
}

uint16_t NavmeshTriangulation::getOriginRegion() const {
  return originRegionId_;
}

math::Vector3 NavmeshTriangulation::transformRegionPointIntoAbsolute(const math::Vector3 &point, const uint16_t regionId) const {
  const auto [regionX, regionY] = sro::position_math::sectorsFromWorldRegionId(regionId);
  const auto [originRegionX, originRegionY] = sro::position_math::sectorsFromWorldRegionId(originRegionId_);
  const auto absoluteX = point.x + (regionX-originRegionX)*1920.0f;
  const auto absoluteZ = point.z + (regionY-originRegionY)*1920.0f;
  return math::Vector3{absoluteX, point.y, absoluteZ};
}

std::pair<uint16_t, math::Vector3> NavmeshTriangulation::transformAbsolutePointIntoRegion(const math::Vector3 &point) const {
  const auto [originRegionX, originRegionY] = sro::position_math::sectorsFromWorldRegionId(originRegionId_);
  const int regionXOffset = static_cast<int>(std::floor(point.x/1920.0));
  const int regionYOffset = static_cast<int>(std::floor(point.z/1920.0));
  const auto newRegionId = sro::position_math::worldRegionIdFromSectors(originRegionX+regionXOffset, originRegionY+regionYOffset);
  math::Vector3 newPoint{point.x-(1920.0f*regionXOffset), point.y, point.z-(1920.0f*regionYOffset)};
  return {newRegionId, newPoint};
}

const NavmeshTriangulation::NavmeshTriangulationMapType& NavmeshTriangulation::getNavmeshTriangulationMap() const {
  return navmeshTriangulationMap_;
}

std::optional<NavmeshTriangulation::IndexType> NavmeshTriangulation::getVertexIndex(const pathfinder::Vector &vertex) const {
  const auto [regionId, regionVertex] = translatePointToRegion(vertex);
  if (!regionExists(regionId)) {
    return {};
  }
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto regionVertexIndex = regionTriangulation.getVertexIndex(regionVertex);
  if (!regionVertexIndex.has_value()) {
    return {};
  }
  return createIndex(*regionVertexIndex, regionId);
}

std::optional<NavmeshTriangulation::IndexType> NavmeshTriangulation::findTriangleForPoint(const pathfinder::Vector &point) const {
  const auto [regionId, regionPoint] = translatePointToRegion(point);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  auto triangleForPoint = regionTriangulation.findTriangleForPoint(regionPoint);
  if (!triangleForPoint.has_value()) {
    // Couldn't find triangle
    return {};
  }
  // Have valid point
  return createIndex(*triangleForPoint, regionId);
}

NavmeshTriangulation::State NavmeshTriangulation::createStartState(const math::Vector3 &point, const IndexType triangleIndex) const {
  const auto [regionId, index] = splitRegionAndIndex(triangleIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto regionIdPointPair = transformAbsolutePointIntoRegion(point);
  const auto regionState = regionTriangulation.createStartState(regionIdPointPair.second, triangleIndex);
  return createGlobalState(regionState, regionId);
}

NavmeshTriangulation::State NavmeshTriangulation::createGoalState(const math::Vector3 &point, const IndexType triangleIndex) const {
  const auto [regionId, index] = splitRegionAndIndex(triangleIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto regionIdPointPair = transformAbsolutePointIntoRegion(point);
  const auto regionState = regionTriangulation.createGoalState(regionIdPointPair.second, triangleIndex);
  return createGlobalState(regionState, regionId);
}

NavmeshTriangulation::TriangleVertexIndicesType NavmeshTriangulation::getTriangleVertexIndices(const IndexType triangleIndex) const {
  const auto [regionId, index] = splitRegionAndIndex(triangleIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto [vertexIndex1, vertexIndex2, vertexIndex3] = regionTriangulation.getTriangleVertexIndices(index);
  return {createIndex(vertexIndex1, regionId), createIndex(vertexIndex2, regionId), createIndex(vertexIndex3, regionId)};
}

NavmeshTriangulation::TriangleEdgeIndicesType NavmeshTriangulation::getTriangleEdgeIndices(const IndexType triangleIndex) const {
  const auto [regionId, index] = splitRegionAndIndex(triangleIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto [edgeIndex1, edgeIndex2, edgeIndex3] = regionTriangulation.getTriangleEdgeIndices(index);
  return {createIndex(edgeIndex1, regionId), createIndex(edgeIndex2, regionId), createIndex(edgeIndex3, regionId)};
}

NavmeshTriangulation::MarkerType NavmeshTriangulation::getVertexMarker(const IndexType vertexIndex) const {
  const auto [regionId, index] = splitRegionAndIndex(vertexIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  return regionTriangulation.getVertexMarker(index);
}

pathfinder::Vector NavmeshTriangulation::getVertex(const IndexType vertexIndex) const {
  const auto [regionId, index] = splitRegionAndIndex(vertexIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto regionVertex = regionTriangulation.getVertex(index);
  return translatePointToGlobal(regionVertex, regionId);
}

NavmeshTriangulation::TriangleVerticesType NavmeshTriangulation::getTriangleVertices(const IndexType triangleIndex) const {
  const auto [regionId, index] = splitRegionAndIndex(triangleIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto [triangleVertex1, triangleVertex2, triangleVertex3] = regionTriangulation.getTriangleVertices(index);
  return {translatePointToGlobal(triangleVertex1, regionId), translatePointToGlobal(triangleVertex2, regionId), translatePointToGlobal(triangleVertex3, regionId)};
}

std::optional<NavmeshTriangulation::GlobalEdgeAndTriangleIndices> NavmeshTriangulation::getNeighborTriangleAndEdge(const IndexType edgeIndex) const {
  auto getNeighborRegionIdAcrossEdge = [](const auto edge, const uint16_t regionId) {
    auto [neighborRegionX, neighborRegionY] = sro::position_math::sectorsFromWorldRegionId(regionId);
    if (edge.first.x() == edge.second.x()) {
      // Edge is vertical
      if (edge.first.x() == 0.0) {
        // Is the left edge of this region
        --neighborRegionX;
      } else if (edge.first.x() == 1920.0) {
        // Is the right edge of this region
        ++neighborRegionX;
      } else {
        throw std::runtime_error("Vertical global edge is neither on the left nor right of the region");
      }
    } else if (edge.first.y() == edge.second.y()) {
      // Edge is horizontal
      if (edge.first.y() == 0.0) {
        // Is the bottom edge of this region
        --neighborRegionY;
      } else if (edge.first.y() == 1920.0) {
        // Is the top edge of this region
        ++neighborRegionY;
      } else {
        throw std::runtime_error("Horizontal global edge is neither on the left nor right of the region");
      }
    } else {
      throw std::runtime_error("Global edge is neither vertical nor horizontal");
    }

    return sro::position_math::worldRegionIdFromSectors(neighborRegionX, neighborRegionY);
  };

  // Get the neighboring region ID
  const auto [regionId, localEdgeIndex] = splitRegionAndIndex(edgeIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto edge = regionTriangulation.getEdge(localEdgeIndex);
  const auto neighborRegionId = getNeighborRegionIdAcrossEdge(edge, regionId);

  // Check if the neighboring region exists
  auto neighborRegionIt = navmeshTriangulationMap_.find(neighborRegionId);
  if (neighborRegionIt == navmeshTriangulationMap_.end()) {
    // This region does not exist
    return {};
  }

  // Get the edge and triangle in the neighboring region (precomputed data)
  const auto globalEdgeLinkIt = globalEdgeAndTriangleLinkMap_.find(edgeIndex);
  if (globalEdgeLinkIt == globalEdgeAndTriangleLinkMap_.end()) {
    throw std::runtime_error("No link exists for this edge");
  }
  return globalEdgeLinkIt->second;
}

std::vector<NavmeshTriangulation::State> NavmeshTriangulation::getSuccessors(const State &currentState, const std::optional<State> goalState, const double agentRadius) const {
  if (currentState.isGoal()) {
    throw std::runtime_error("Trying to get successors of goal");
  }

  if (goalState.has_value()) {
    if (currentState.isSameTriangleAs(*goalState)) {
      // This is the goal, only successor is the goal point itself
      auto newGoalState{currentState};
      newGoalState.setIsGoal(true);
      return {newGoalState};
    }
  }

  // Get successors from single region
  const auto [currentStateRegionId, currentStateTriangleIndex] = splitRegionAndIndex(currentState.getTriangleIndex());
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(currentStateRegionId);
  const auto regionCurrentState = createRegionState(currentState);
  std::optional<SingleRegionNavmeshTriangulation::State> optionalGoal;
  if (goalState.has_value()) {
    const auto [goalTriangleRegionId, goalTriangleRegionIndex] = splitRegionAndIndex(goalState->getTriangleIndex());
    const auto regionGoalState = createRegionState(*goalState);
    if (goalTriangleRegionId == currentStateRegionId) {
      // State and goal are in same region, we can pass the goal state to getSuccessors
      optionalGoal = regionGoalState;
    }
  }
  std::vector<SingleRegionNavmeshTriangulation::State> regionSuccessors = regionTriangulation.getSuccessors(regionCurrentState, optionalGoal, agentRadius);

  std::vector<State> globalSuccessors;
  std::transform(regionSuccessors.begin(), regionSuccessors.end(), std::back_inserter(globalSuccessors), [currentStateRegionId = currentStateRegionId](const auto &regionCurrentState) {
    return createGlobalState(regionCurrentState, currentStateRegionId);
  });

  // Try to see if there are any additional successors that leave the region from the currentState
  const auto [currentTriangleEdgeIndex0, currentTriangleEdgeIndex1, currentTriangleEdgeIndex2] = regionTriangulation.getTriangleEdgeIndices(currentStateTriangleIndex);
  // ==================================================================
  for (const auto currentTriangleEdgeIndex : {currentTriangleEdgeIndex0, currentTriangleEdgeIndex1, currentTriangleEdgeIndex2}) {
    if (regionCurrentState.hasEntryEdgeIndex() && regionCurrentState.getEntryEdgeIndex() == currentTriangleEdgeIndex) {
      // Entered through this edge, dont return
      // Note: This might require that the "entry edge" of a state is always the edge from the same region as the triangle of the state
      continue;
    }
    const auto edgeMarker = regionTriangulation.getEdgeMarker(currentTriangleEdgeIndex);
    if (edgeMarker >= 2) {
      // Edge is some kind of constraint (might not necessarily be blocking)
      const auto edgeConstraintData = regionTriangulation.getEdgeConstraintData(edgeMarker)[0]; //TODO: TODO! TODO! TODO! TODO!
      if (edgeConstraintData.forTerrain() && edgeConstraintData.is(EdgeConstraintFlag::kGlobal)) {
        // This is a global edge (boundary between regions)
        if (!edgeConstraintData.is(EdgeConstraintFlag::kBlocking) || currentState.isOnObject()) {
          // Only pass through a non-blocking edge if we're not on an object
          if (regionTriangulation.agentFitsThroughEdge(currentTriangleEdgeIndex, agentRadius)) {
            // Try to cross into the next region
            const auto globalEdgeIndex = createIndex(currentTriangleEdgeIndex, currentStateRegionId);
            const auto neighboringRegionTriangleAndEdge = getNeighborTriangleAndEdge(globalEdgeIndex);
            if (neighboringRegionTriangleAndEdge.has_value()) {
              State stateInNeighboringRegion{currentState};
              stateInNeighboringRegion.setNewTriangleAndEntryEdge(neighboringRegionTriangleAndEdge->triangleIndex, neighboringRegionTriangleAndEdge->edgeIndex);
              // std::cout << "Creating new global successor state! " << stateInNeighboringRegion << std::endl;
              globalSuccessors.push_back(stateInNeighboringRegion);
            }
          }
        }
      }
    }
  }
  return globalSuccessors;
}

std::vector<NavmeshTriangulation::State> NavmeshTriangulation::getNeighborsInObjectArea(const State &currentState) const {
  // Get successors from single region
  const auto [currentStateRegionId, currentStateTriangleIndex] = splitRegionAndIndex(currentState.getTriangleIndex());
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(currentStateRegionId);
  const auto regionCurrentState = createRegionState(currentState);
  std::vector<SingleRegionNavmeshTriangulation::State> regionSuccessors = regionTriangulation.getNeighborsInObjectArea(regionCurrentState);

  std::vector<State> globalSuccessors;
  std::transform(regionSuccessors.begin(), regionSuccessors.end(), std::back_inserter(globalSuccessors), [currentStateRegionId = currentStateRegionId](const auto &regionCurrentState) {
    return createGlobalState(regionCurrentState, currentStateRegionId);
  });
  // Try to see if there are any additional successors that leave the region from the currentState
  const auto [currentTriangleEdgeIndex0, currentTriangleEdgeIndex1, currentTriangleEdgeIndex2] = regionTriangulation.getTriangleEdgeIndices(currentStateTriangleIndex);
  for (const auto currentTriangleEdgeIndex : {currentTriangleEdgeIndex0, currentTriangleEdgeIndex1, currentTriangleEdgeIndex2}) {
    if (regionCurrentState.hasEntryEdgeIndex() && regionCurrentState.getEntryEdgeIndex() == currentTriangleEdgeIndex) {
      // Entered through this edge, dont return
      // Note: This might require that the "entry edge" of a state is always the edge from the same region as the triangle of the state
      continue;
    }
    const auto edgeMarker = regionTriangulation.getEdgeMarker(currentTriangleEdgeIndex);
    if (edgeMarker >= 2) {
      // Edge is some kind of constraint (might not necessarily be blocking)
      const auto edgeConstraintData = regionTriangulation.getEdgeConstraintData(edgeMarker)[0]; //TODO: TODO! TODO! TODO! TODO!
      if (edgeConstraintData.forTerrain() && edgeConstraintData.is(EdgeConstraintFlag::kGlobal)) {
        // This is a global edge (boundary between regions)
        if (!edgeConstraintData.is(EdgeConstraintFlag::kBlocking) || currentState.isOnObject()) {
          // Only pass through a non-blocking edge if we're not on an object
          // Try to cross into the next region
          const auto globalEdgeIndex = createIndex(currentTriangleEdgeIndex, currentStateRegionId);
          const auto neighboringRegionTriangleAndEdge = getNeighborTriangleAndEdge(globalEdgeIndex);
          if (neighboringRegionTriangleAndEdge.has_value()) {
            State stateInNeighboringRegion{currentState};
            stateInNeighboringRegion.setNewTriangleAndEntryEdge(neighboringRegionTriangleAndEdge->triangleIndex, neighboringRegionTriangleAndEdge->edgeIndex);
            // std::cout << "Creating new global successor state! " << stateInNeighboringRegion << std::endl;
            globalSuccessors.push_back(stateInNeighboringRegion);
          }
        }
      }
    }
  }
  return globalSuccessors;
}

NavmeshTriangulation::EdgeType NavmeshTriangulation::getSharedEdge(const IndexType triangle1Index, const IndexType triangle2Index) const {
  const auto [triangle1RegionId, triangle1SingleRegionIndex] = splitRegionAndIndex(triangle1Index);
  const auto [triangle2RegionId, triangle2SingleRegionIndex] = splitRegionAndIndex(triangle2Index);
  if (triangle1RegionId == triangle2RegionId) {
    // Within same region
    const auto &regionTriangulation = getNavmeshTriangulationForRegion(triangle1RegionId);
    const auto [edgeVertex1, edgeVertex2] = regionTriangulation.getSharedEdge(triangle1SingleRegionIndex, triangle2SingleRegionIndex);
    return {translatePointToGlobal(edgeVertex1, triangle1RegionId), translatePointToGlobal(edgeVertex2, triangle1RegionId)};
  } else {
    // They're in different regions
    // Quick check to make sure that they're in neighboring regions
    const auto [triangle1RegionX, triangle1RegionY] = sro::position_math::sectorsFromWorldRegionId(triangle1RegionId);
    const auto [triangle2RegionX, triangle2RegionY] = sro::position_math::sectorsFromWorldRegionId(triangle2RegionId);
    if (std::abs(triangle1RegionX-triangle2RegionX)+std::abs(triangle1RegionY-triangle2RegionY) != 1) {
      throw std::runtime_error("Trying to get shared edge between two regions which are not neighbors");
    }
    std::optional<IndexType> sharedEdgeGlobalIndex;
    const auto [edgeIndex0, edgeIndex1, edgeIndex2] = getTriangleEdgeIndices(triangle2Index);
    for (const auto edgeIndex : {edgeIndex0, edgeIndex1, edgeIndex2}) {
      const auto globalEdgeLinkIt = globalEdgeAndTriangleLinkMap_.find(edgeIndex);
      if (globalEdgeLinkIt != globalEdgeAndTriangleLinkMap_.end()) {
        if (globalEdgeLinkIt->second.triangleIndex == triangle1Index) {
          // `edgeIndex` is the edge (on triangle2Index's side) that is shared between these two triangles
          if (sharedEdgeGlobalIndex.has_value()) {
            throw std::runtime_error("These triangles have multiple shared edges");
          }
          sharedEdgeGlobalIndex = edgeIndex;
        }
      }
    }
    if (!sharedEdgeGlobalIndex.has_value()) {
      throw std::runtime_error("These triangles do not have a shared edge");
    }
    return getEdge(*sharedEdgeGlobalIndex);
  }
}

NavmeshTriangulation::EdgeType NavmeshTriangulation::getEdge(const IndexType edgeIndex) const {
  const auto [regionId, index] = splitRegionAndIndex(edgeIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto [edgeVertex1, edgeVertex2] = regionTriangulation.getEdge(index);
  return {translatePointToGlobal(edgeVertex1, regionId), translatePointToGlobal(edgeVertex2, regionId)};
}

NavmeshTriangulation::EdgeVertexIndicesType NavmeshTriangulation::getEdgeVertexIndices(const IndexType edgeIndex) const {
  const auto [regionId, index] = splitRegionAndIndex(edgeIndex);
  const auto &regionTriangulation = getNavmeshTriangulationForRegion(regionId);
  const auto [edgeIndex1, edgeIndex2] = regionTriangulation.getEdgeVertexIndices(index);
  return {createIndex(edgeIndex1, regionId), createIndex(edgeIndex2, regionId)};
}
  
pathfinder::Vector NavmeshTriangulation::translatePointToGlobal(pathfinder::Vector point, const uint16_t regionId) const {
  const auto [originRegionX, originRegionY] = sro::position_math::sectorsFromWorldRegionId(originRegionId_);
  const auto [regionX, regionY] = sro::position_math::sectorsFromWorldRegionId(regionId);
  point.setX(point.x() + 1920.0 * (regionX-originRegionX));
  point.setY(point.y() + 1920.0 * (regionY-originRegionY));
  return point;
}

std::pair<uint16_t,pathfinder::Vector> NavmeshTriangulation::translatePointToRegion(const pathfinder::Vector &point) const {
  const auto [originRegionX, originRegionY] = sro::position_math::sectorsFromWorldRegionId(originRegionId_);
  const int regionXOffset = static_cast<int>(std::floor(point.x()/1920.0));
  const int regionYOffset = static_cast<int>(std::floor(point.y()/1920.0));
  const auto newRegionId = sro::position_math::worldRegionIdFromSectors(originRegionX+regionXOffset, originRegionY+regionYOffset);
  pathfinder::Vector newPoint{point.x()-(1920.0*regionXOffset), point.y()-(1920.0*regionYOffset)};
  return {newRegionId, newPoint};
}

bool NavmeshTriangulation::regionExists(const uint16_t regionId) const {
  return (navmeshTriangulationMap_.find(regionId) != navmeshTriangulationMap_.end());
}

std::pair<uint16_t, SingleRegionNavmeshTriangulation::IndexType> NavmeshTriangulation::splitRegionAndIndex(const IndexType index) {
  return {static_cast<uint16_t>(index>>16), static_cast<SingleRegionNavmeshTriangulation::IndexType>(index&0xFFFF)};
}
  
NavmeshTriangulation::IndexType NavmeshTriangulation::createIndex(const SingleRegionNavmeshTriangulation::IndexType index, const uint16_t regionId) {
  return ((static_cast<NavmeshTriangulation::IndexType>(regionId)<<16) | index);
}

SingleRegionNavmeshTriangulation::State NavmeshTriangulation::createRegionState(const State &globalState) {
  auto constructState = [&]() -> SingleRegionNavmeshTriangulation::State {
    const auto [regionId, triangleIndex] = splitRegionAndIndex(globalState.getTriangleIndex());
    if (globalState.hasEntryEdgeIndex()) {
      return SingleRegionNavmeshTriangulation::State(triangleIndex, splitRegionAndIndex(globalState.getEntryEdgeIndex()).second);
    } else {
      return SingleRegionNavmeshTriangulation::State(triangleIndex);
    }
  };

  SingleRegionNavmeshTriangulation::State regionState = constructState();
  regionState.setIsGoal(globalState.isGoal());
  if (globalState.isOnObject()) {
    regionState.setObjectData(globalState.getObjectData());
  } else {
    regionState.setOnTerrain();
  }
  if (globalState.isTraversingLink()) {
    regionState.setLinkId(globalState.getLinkId());
  } else {
    regionState.resetLinkId();
  }
  return regionState;
}

NavmeshTriangulation::State NavmeshTriangulation::createGlobalState(const SingleRegionNavmeshTriangulation::State &regionState, const uint16_t regionId) {
  auto constructState = [&]() -> NavmeshTriangulation::State {
    const auto triangleIndex = createIndex(regionState.getTriangleIndex(), regionId);
    if (regionState.hasEntryEdgeIndex()) {
      return NavmeshTriangulation::State(triangleIndex, createIndex(regionState.getEntryEdgeIndex(), regionId));
    } else {
      return NavmeshTriangulation::State(triangleIndex);
    }
  };

  NavmeshTriangulation::State globalState = constructState();
  globalState.setIsGoal(regionState.isGoal());
  if (regionState.isOnObject()) {
    globalState.setObjectData(regionState.getObjectData());
  } else {
    globalState.setOnTerrain();
  }
  if (regionState.isTraversingLink()) {
    globalState.setLinkId(regionState.getLinkId());
  } else {
    globalState.resetLinkId();
  }
  return globalState;
}

pathfinder::Vector NavmeshTriangulation::to2dPoint(const math::Vector3 &point) {
  // Convert our 3d point into the pathfinder's 2d point type
  return {point.x, point.z};
}

// =========================================================================================================
// =========================================================================================================
// =========================================================================================================

void NavmeshTriangulation::buildGlobalEdgesBasedOnBlockedTerrain(const Navmesh &navmesh, const Region &region, std::vector<navmesh::Edge> &globalEdges) {
  // We want to minimize the number of global edges of this region while retaining all traversability information
  enum class GlobalEdgeSide {
    kLeft,
    kRight,
    kBottom,
    kTop,
    kNone
  };

  auto getNeighboringRegionId = [](const uint16_t regionId, const GlobalEdgeSide side) {
    const auto [regionX, regionY] = sro::position_math::sectorsFromWorldRegionId(regionId);
    if (side == GlobalEdgeSide::kLeft) {
      return sro::position_math::worldRegionIdFromSectors(regionX-1,regionY);
    } else if (side == GlobalEdgeSide::kRight) {
      return sro::position_math::worldRegionIdFromSectors(regionX+1,regionY);
    } else if (side == GlobalEdgeSide::kBottom) {
      return sro::position_math::worldRegionIdFromSectors(regionX,regionY-1);
    } else if (side == GlobalEdgeSide::kTop) {
      return sro::position_math::worldRegionIdFromSectors(regionX,regionY+1);
    } else {
      throw std::runtime_error("Impossible direction");
    }
  };

  auto getGlobalEdgeSide = [](const auto &edge) {
    if (edge.min.x == edge.max.x) {
      if (edge.min.x == 0.0) {
        return GlobalEdgeSide::kLeft;
      } else if (edge.min.x == 1920.0) {
        return GlobalEdgeSide::kRight;
      }
    } else if (edge.min.z == edge.max.z) {
      if (edge.min.z == 0.0) {
        return GlobalEdgeSide::kBottom;
      } else if (edge.min.z == 1920.0) {
        return GlobalEdgeSide::kTop;
      }
    }
    return GlobalEdgeSide::kNone;
  };

  // I think that the only real reason to have multiple edges on the boundary of the region is to account for blocked tiles in our region or in the neighboring region
  enum class State {
    kNeitherTilesEnabled = 0,
    kOurTileEnabled = 1,
    kNeighborTileEnabled = 2,
    kBothTilesEnabled = 3
  };
  auto stateToString = [](const State s) -> std::string {
    if (s == State::kNeitherTilesEnabled) {
      return "kNeitherTilesEnabled";
    } else if (s == State::kOurTileEnabled) {
      return "kOurTileEnabled";
    } else if (s == State::kNeighborTileEnabled) {
      return "kNeighborTileEnabled";
    } else if (s == State::kBothTilesEnabled) {
      return "kBothTilesEnabled";
    } else {
      throw std::runtime_error("Impossible state");
    }
  };

  const auto leftRegionId = getNeighboringRegionId(region.id, GlobalEdgeSide::kLeft);
  const auto rightRegionId = getNeighboringRegionId(region.id, GlobalEdgeSide::kRight);
  const auto bottomRegionId = getNeighboringRegionId(region.id, GlobalEdgeSide::kBottom);
  const auto topRegionId = getNeighboringRegionId(region.id, GlobalEdgeSide::kTop);
  const Region *leftRegion{nullptr};
  const Region *rightRegion{nullptr};
  const Region *bottomRegion{nullptr};
  const Region *topRegion{nullptr};
  if (navmesh.regionIsEnabled(leftRegionId)) {
    leftRegion = &navmesh.getRegion(leftRegionId);
  }
  if (navmesh.regionIsEnabled(rightRegionId)) {
    rightRegion = &navmesh.getRegion(rightRegionId);
  }
  if (navmesh.regionIsEnabled(bottomRegionId)) {
    bottomRegion = &navmesh.getRegion(bottomRegionId);
  }
  if (navmesh.regionIsEnabled(topRegionId)) {
    topRegion = &navmesh.getRegion(topRegionId);
  }

  auto getStateOfBlockedTilesAtBoundary = [](const auto &ourRegion, const int ourRow, const int ourCol, const auto *neighborRegion, const int neighborRow, const int neighborCol) -> State {
    State currentState = State::kNeitherTilesEnabled;
    if (ourRegion.enabledTiles[ourRow][ourCol]) {
      currentState = static_cast<State>(static_cast<uint32_t>(currentState) | static_cast<uint32_t>(State::kOurTileEnabled));
    }
    if (neighborRegion != nullptr && neighborRegion->enabledTiles[neighborRow][neighborCol]) {
      currentState = static_cast<State>(static_cast<uint32_t>(currentState) | static_cast<uint32_t>(State::kNeighborTileEnabled));
    }
    return currentState;
  };

  auto addEdge = [&globalEdges](const math::Vector3 &p1, const math::Vector3 &p2, const State state) {
    navmesh::Edge edge;
    edge.min = p1;
    edge.max = p2;
    if (state == State::kBothTilesEnabled) {
      edge.flag = EdgeFlag::kNone;
    } else {
      edge.flag = EdgeFlag::kBlocked;
    }
    globalEdges.push_back(edge);
  };

  constexpr const int kTileCount = std::tuple_size_v<decltype(region.enabledTiles)>;

  State lastStateLeft = getStateOfBlockedTilesAtBoundary(region, 0, 0, leftRegion, 0, kTileCount-1);
  State lastStateRight = getStateOfBlockedTilesAtBoundary(region, 0, kTileCount-1, rightRegion, 0, 0);
  State lastStateBottom = getStateOfBlockedTilesAtBoundary(region, 0, 0, bottomRegion, kTileCount-1, 0);
  State lastStateTop = getStateOfBlockedTilesAtBoundary(region, kTileCount-1, 0, topRegion, 0, 0);
  float lastLeftPoint{0.0};
  float lastRightPoint{0.0};
  float lastBottomPoint{0.0};
  float lastTopPoint{0.0};

  for (int row=1; row<kTileCount; ++row) {
    State currentStateLeft = getStateOfBlockedTilesAtBoundary(region, row, 0, leftRegion, row, kTileCount-1);
    if (currentStateLeft != lastStateLeft) {
      const float currentPoint = row * 20.0f;
      addEdge({0.0, 0.0, lastLeftPoint}, {0.0, 0.0, currentPoint}, lastStateLeft);
      lastStateLeft = currentStateLeft;
      lastLeftPoint = currentPoint;
    }

    State currentStateRight = getStateOfBlockedTilesAtBoundary(region, row, kTileCount-1, rightRegion, row, 0);
    if (currentStateRight != lastStateRight) {
      const float currentPoint = row * 20.0f;
      addEdge({1920.0, 0.0, lastRightPoint}, {1920.0, 0.0, currentPoint}, lastStateRight);
      lastStateRight = currentStateRight;
      lastRightPoint = currentPoint;
    }
  }
  addEdge({0.0, 0.0, lastLeftPoint}, {0.0, 0.0, 1920.0}, lastStateLeft);
  addEdge({1920.0, 0.0, lastRightPoint}, {1920.0, 0.0, 1920.0}, lastStateRight);

  for (int col=1; col<kTileCount; ++col) {
    State currentStateBottom = getStateOfBlockedTilesAtBoundary(region, 0, col, bottomRegion, kTileCount-1, col);
    if (currentStateBottom != lastStateBottom) {
      const float currentPoint = col * 20.0f;
      addEdge({lastBottomPoint, 0.0, 0.0}, {currentPoint, 0.0, 0.0}, lastStateBottom);
      lastStateBottom = currentStateBottom;
      lastBottomPoint = currentPoint;
    }

    State currentStateTop = getStateOfBlockedTilesAtBoundary(region, kTileCount-1, col, topRegion, 0, col);
    if (currentStateTop != lastStateTop) {
      const float currentPoint = col * 20.0f;
      addEdge({lastTopPoint, 0.0, 1920.0}, {currentPoint, 0.0, 1920.0}, lastStateTop);
      lastStateTop = currentStateTop;
      lastTopPoint = currentPoint;
    }
  }
  addEdge({lastBottomPoint, 0.0, 0.0}, {1920.0, 0.0, 0.0}, lastStateBottom);
  addEdge({lastTopPoint, 0.0, 1920.0}, {1920.0, 0.0, 1920.0}, lastStateTop);
}

void NavmeshTriangulation::buildNavmeshForRegion(const Navmesh &navmesh, const Region &region) {
  using NavmeshIndexType = SingleRegionNavmeshTriangulation::IndexType;
  using PointListType = std::vector<pathfinder::Vector>;
  struct EdgeType {
    EdgeType(NavmeshIndexType a, NavmeshIndexType b, int c) : vertex0(a), vertex1(b), marker(c) {}
    NavmeshIndexType vertex0, vertex1;
    // Marker 0 is reserved for no marker
    // Marker 1 is reserved for boundary of triangulation
    int marker;
  };

  using EdgeListType = std::vector<EdgeType>;

  std::vector<std::vector<ConstraintData>> constraintData;
  int nextMarker=2;
  auto addConstraintDataAndGetMarker = [&](const std::vector<ConstraintData> &constraints) {
    // TODO: We could consolidate markers
    //  Actually, we are starting to rely on them being unique per edge; maybe not
    constraintData.push_back(constraints);
    return nextMarker++;
  };

  auto createLinkAndGetId = [this](const auto &link) -> std::pair<uint32_t,bool> {
    // Does this link already exist?
    // TODO: Does it make sense to spend this extra time, or are duplicates fine?
    for (int i=0; i<linkData_.size(); ++i) {
      const auto &otherLink = linkData_[i];
      if (otherLink.srcObjectGlobalId == link.srcObjectGlobalId &&
          otherLink.destObjectGlobalId == link.destObjectGlobalId &&
          otherLink.srcEdgeIndex == link.srcEdgeIndex &&
          otherLink.destEdgeIndex == link.destEdgeIndex) {
        // This link exists and it exactly matches
        return {i, true};
      } else if (otherLink.srcObjectGlobalId == link.destObjectGlobalId &&
          otherLink.destObjectGlobalId == link.srcObjectGlobalId &&
          otherLink.srcEdgeIndex == link.destEdgeIndex &&
          otherLink.destEdgeIndex == link.srcEdgeIndex) {
        // This link exists but is flipped
        return {i, false};
      }
    }

    // This link does not yet exist
    linkData_.push_back(link);
    return {linkData_.size()-1, true};
  };

  // ============================Lambdas============================
  // Now, extract data from the navmesh
  auto addVertexAndGetIndex = [](const math::Vector3 &p, PointListType &points) -> size_t {
    // TODO: Maybe give a tiny range for precision error
    auto it = std::find_if(points.begin(), points.end(), [&p](const pathfinder::Vector &otherPoint){
      constexpr const double kReducedPrecision{1e-3};
      return (pathfinder::math::equal(otherPoint.x(), p.x/* , kReducedPrecision */) && pathfinder::math::equal(otherPoint.y(), p.z/* , kReducedPrecision */));
    });
    if (it != points.end()) {
      // Point already exists in list
      // std::cout << "Vertex already exists! " << p.x << ',' << p.z << " at index " << std::distance(points.begin(), it) << '\n';
      return std::distance(points.begin(), it);
    }
    // std::cout << "Insert vertex " << p.x << ',' << p.z << " at index " << points.size() << '\n';
    points.emplace_back(p.x, p.z);
    return points.size()-1;
  };

  auto addEdge = [&](math::Vector3 v1, math::Vector3 v2, const ConstraintData &constraint, PointListType &points, EdgeListType &edges) {
    if (v1.x == v2.x && v1.z == v2.z) {
      // There are some edges which are strictly vertical (only differ by y-value/height)
      // We will ignore those since we are only working in 2d
      return;
    }

    // Start by sorting these vertices
    //  This ordering will be used later when checking for overlaps
    if (v1.x == v2.x) {
      if (v1.z > v2.z) {
        std::swap(v1,v2);
      }
    } else if (v1.x > v2.x) {
      std::swap(v1, v2);
    }

    // Create vertices for this edge
    const auto v1Index = addVertexAndGetIndex(v1, points);
    const auto v2Index = addVertexAndGetIndex(v2, points);

    // Check if this edge already exists
    auto it = std::find_if(edges.begin(), edges.end(), [v1Index, v2Index](const EdgeType &edge) {
      return (edge.vertex0 == v1Index && edge.vertex1 == v2Index) || (edge.vertex0 == v2Index && edge.vertex1 == v1Index);
    });

    if (it == edges.end()) {
      // Creating an edge which does not exist
      // Lets check if this edge overlaps with any other edge
      // In the case when it does overlap with another edge, we need to add a new constraint to a subset of the edge that we overlap with

      auto vectorLess = [](auto &v1, auto &v2) {
        if (v1.x() == v2.x()) {
          return v1.y() < v2.y();
        } else {
          return v1.x() < v2.x();
        }
      };

      pathfinder::Vector edgeVertex0(v1.x, v1.z);
      pathfinder::Vector edgeVertex1(v2.x, v2.z);

      std::vector<int> indicesOfOverlappingEdges;
      for (int i=0; i<edges.size(); ++i) {
        const auto &otherEdge = edges.at(i);
        const auto otherEdgeVertex0 = points.at(otherEdge.vertex0);
        const auto otherEdgeVertex1 = points.at(otherEdge.vertex1);
        const auto res1 = pathfinder::math::crossProductForSign(otherEdgeVertex0, otherEdgeVertex1, otherEdgeVertex0, edgeVertex0);
        const auto res2 = pathfinder::math::crossProductForSign(otherEdgeVertex0, otherEdgeVertex1, otherEdgeVertex0, edgeVertex1);
        if (pathfinder::math::equal(res1, 0.0) && pathfinder::math::equal(res2, 0.0)) {
          // These two line segments are on the same line
          // There are 6 cases we care about:
          //  A1. The new edge overlaps and extends beyond our edge
          //  A2. Our edge is completely within the new edge (not sharing endpoints)
          //  A3. Our edge is completely within the new edge (sharing one endpoint)
          //  A4. The new edge is completely within our edge (not sharing endpoints)
          //  A5. The new edge is completely within our edge (sharing one endpoint)
          //  A6. The new edge is completely within our edge (sharing both endpoints)
          //    This is the same as two edges being the same; this has already been handled
          // There are 2 other cases that are possible, but not an overlap:
          //  B1. The two edges do not overlap at all
          //  B2. The two edges share only a single endpoint

          // Check if any vertex is between the two vertices of the other line
          if ((vectorLess(otherEdgeVertex0, edgeVertex0) && vectorLess(edgeVertex0, otherEdgeVertex1)) ||
              (vectorLess(otherEdgeVertex0, edgeVertex1) && vectorLess(edgeVertex1, otherEdgeVertex1)) ||
              (vectorLess(edgeVertex0, otherEdgeVertex0) && vectorLess(otherEdgeVertex0, edgeVertex1)) ||
              (vectorLess(edgeVertex0, otherEdgeVertex1) && vectorLess(otherEdgeVertex1, edgeVertex1))) {
            // There is an actual overlap
            // Note: This trusts that the two endpoints of every edge are sorted
            const auto &otherEdgeConstraints = constraintData.at(otherEdge.marker-2);
            if (std::find(otherEdgeConstraints.begin(), otherEdgeConstraints.end(), constraint) == otherEdgeConstraints.end()) {
              // Different constraints
              indicesOfOverlappingEdges.push_back(i);
            } else {
              // Even though these edges seem to overlap, their constraints are the same, we dont really care in this case
              // TODO: If we were ambitious, we could consolidate these two lines into one
              //  i.e. A------->B
              //            C--------->D
              //  becomes
              //       A-------------->D
            }
          }
        }
      }

      if (indicesOfOverlappingEdges.empty()) {
        // No overlap, lets just add our edge and be done
        const auto marker = addConstraintDataAndGetMarker({constraint});
        edges.emplace_back(static_cast<NavmeshIndexType>(v1Index), static_cast<NavmeshIndexType>(v2Index), marker);
        return;
      }

      // Fact: There is overlap between these edges
      struct EdgeVertex {
        int vertexIndex;
        bool isStart;
        int edgeIndex;
      };
      std::vector<EdgeVertex> edgePoints;
      // Add both points of our edge to the list
      edgePoints.push_back({static_cast<int>(v1Index), true, -1});
      edgePoints.push_back({static_cast<int>(v2Index), false, -1});

      // Add both points of all of the other edges to the list
      for (const auto i : indicesOfOverlappingEdges) {
        const auto &otherEdge = edges.at(i);
        edgePoints.push_back({otherEdge.vertex0, true, i});
        edgePoints.push_back({otherEdge.vertex1, false, i});
      }
      // Sort points
      std::sort(edgePoints.begin(), edgePoints.end(), [&points,&vectorLess](const auto &e1, const auto &e2) {
        const auto &v1 = points.at(e1.vertexIndex);
        const auto &v2 = points.at(e2.vertexIndex);
        return vectorLess(v1, v2);
      });
      // Keep constraint data for each edge
      std::map<int, std::vector<ConstraintData>> constraintDataMap;
      auto createConstraintDataList = [&constraintDataMap]{
        std::vector<ConstraintData> data;
        for (const auto &indexDataPair : constraintDataMap) {
          data.insert(data.end(), indexDataPair.second.begin(), indexDataPair.second.end());
        }
        return data;
      };

      struct EdgeToCreate {
        size_t v0Index, v1Index;
        std::vector<ConstraintData> constraintDatas;
      };
      std::vector<EdgeToCreate> edgesToCreate;

      // Add constraint data for first point
      if (edgePoints[0].edgeIndex == -1) {
        constraintDataMap[-1] = {constraint};
      } else {
        const auto &otherEdge = edges.at(edgePoints[0].edgeIndex);
        const auto &otherEdgeConstraints = constraintData.at(otherEdge.marker-2);
        constraintDataMap[edgePoints[0].edgeIndex] = otherEdgeConstraints;
      }
      for (int i=1; i<edgePoints.size(); ++i) {
        if (edgePoints[i].vertexIndex != edgePoints[i-1].vertexIndex) {
          // Have a non-zero length edge
          // Create it with the current constraint data
          edgesToCreate.push_back({static_cast<size_t>(edgePoints[i-1].vertexIndex), static_cast<size_t>(edgePoints[i].vertexIndex), createConstraintDataList()});
        }
        if (edgePoints[i].isStart) {
          // Add constraint data to map
          if (edgePoints[i].edgeIndex == -1) {
            constraintDataMap[-1] = {constraint};
          } else {
            const auto &otherEdge = edges.at(edgePoints[i].edgeIndex);
            const auto &otherEdgeConstraints = constraintData.at(otherEdge.marker-2);
            constraintDataMap[edgePoints[i].edgeIndex] = otherEdgeConstraints;
          }
        } else {
          // Remove constraint data from map
          constraintDataMap[edgePoints[i].edgeIndex] = {};
        }
      }

      if (indicesOfOverlappingEdges.size() > edgesToCreate.size()) {
        throw std::runtime_error("We are trying to delete more edges than we're creating");
      }
      int edgeToCreateIndex=0;
      while (edgeToCreateIndex<indicesOfOverlappingEdges.size()) {
        const auto edgeToBeOverwrittenIndex = indicesOfOverlappingEdges[edgeToCreateIndex];
        auto &edgeToBeOverwritten = edges.at(edgeToBeOverwrittenIndex);
        const auto &newEdgeData = edgesToCreate[edgeToCreateIndex];
        // Overwrite edgeToBeOverwritten with newEdgeData
        edgeToBeOverwritten.vertex0 = newEdgeData.v0Index;
        edgeToBeOverwritten.vertex1 = newEdgeData.v1Index;
        // Marker stays the same, update the constraint data pointed to by this marker
        auto &existingConstraintData = constraintData.at(edgeToBeOverwritten.marker-2);
        existingConstraintData = newEdgeData.constraintDatas;
        ++edgeToCreateIndex;
      }
      while (edgeToCreateIndex<edgesToCreate.size()) {
        // Create new edge
        const auto &newEdgeData = edgesToCreate[edgeToCreateIndex];
        const auto marker = addConstraintDataAndGetMarker(newEdgeData.constraintDatas);
        edges.emplace_back(static_cast<NavmeshIndexType>(newEdgeData.v0Index), static_cast<NavmeshIndexType>(newEdgeData.v1Index), marker);
        ++edgeToCreateIndex;
      }
    } else {
      // Edge already exists, try to see if we've already saved this constraint data for this edge
      auto &constraintDataListForThisEdge = constraintData.at(it->marker-2);
      if (std::find(constraintDataListForThisEdge.begin(), constraintDataListForThisEdge.end(), constraint) == constraintDataListForThisEdge.end()) {
        // This is a different constraint than any of the existing constraints for this edge, add it to the list of constraints for this edge
        constraintDataListForThisEdge.push_back(constraint);
      }
    }
  };

  auto addObjEdgeWithTrim = [&addEdge](math::Vector3 v1, math::Vector3 v2, const int regionDx, const int regionDy, const ConstraintData &constraint, PointListType &points, EdgeListType &edges) {
    const double regionMinX = 1920.0*regionDx;
    const double regionMinY = 1920.0*regionDy;
    const double regionMaxX = 1920.0*(regionDx+1);
    const double regionMaxY = 1920.0*(regionDy+1);
    // Trim the line in its region
    bool res = geometry_helpers::lineTrimToRegion(v1, v2, regionMinX, regionMinY, regionMaxX, regionMaxY);
    if (res) {
      // Translate points back to within origin region
      v1.x -= regionMinX;
      v2.x -= regionMinX;
      v1.z -= regionMinY;
      v2.z -= regionMinY;
      addEdge(v1, v2, constraint, points, edges);
    }
  };

  // =============================Data==============================
  PointListType inVertices;
  EdgeListType inEdges;

  auto isRegionBoundaryEdge = [](const auto &edge) {
    if (edge.min.x == edge.max.x) {
      if (edge.min.x == 0.0 || edge.min.x == 1920.0) {
        return true;
      }
    } else if (edge.min.z == edge.max.z) {
      if (edge.min.z == 0.0 || edge.min.z == 1920.0) {
        return true;
      }
    }
    return false;
  };

  {
    // Global edges of region
    // Note: We dont use the given data; we calculate our own global edges based on blocked terrain
    std::vector<navmesh::Edge> globalEdges;
    buildGlobalEdgesBasedOnBlockedTerrain(navmesh, region, globalEdges);
    for (const auto &edge : globalEdges) {
      ConstraintData globalEdgeConstraint;
      globalEdgeConstraint.edgeFlag |= EdgeConstraintFlag::kGlobal;
      if (static_cast<uint8_t>(edge.flag) & static_cast<uint8_t>(EdgeFlag::kBlocked)) {
        globalEdgeConstraint.edgeFlag |= EdgeConstraintFlag::kBlocking;
      }
      addEdge(edge.min, edge.max, globalEdgeConstraint, inVertices, inEdges);    
    }
  }

  // TODO: I think we need to resolve vertices which end up having conflicting boundary markers
  //  This includes vertices which are added for multiple reasons
  //  This includes vertices which are placed onto an existing line segment
  //    For example, when an object constraint overlaps with the global edge

  // Interior edges of region
  for (const auto &edge : region.internalEdges) {
    // Skip "internal" edges which are in the global position
    if (!isRegionBoundaryEdge(edge)) {
      // I dont think we care about internal edges which are non-blocking
      // Observing all data, the flag is either kBlockSrc2Dst or kInternal
      // TODO:
      //  For now, we wont care about the directionality of the blocking edge.
      //  This limits us in the case if we somehow end up inside of a blocked (blocked to enter) region of the terrain. We would think that we couldn't exit, but we could.
      //  In order to solve this, we'd have to store enough data to know which cell we're inside at any time
      if (static_cast<uint8_t>(edge.flag) & static_cast<uint8_t>(EdgeFlag::kBlocked)) {
        ConstraintData constraint;
        constraint.edgeFlag |= EdgeConstraintFlag::kInternal;
        constraint.edgeFlag |= EdgeConstraintFlag::kBlocking;
        addEdge(edge.min, edge.max, constraint, inVertices, inEdges);
      }
    }
  }

  // Get contstraining edges for object
  for (const auto &objectInstanceId : region.objectInstanceIds) {
    // Transform the object into its owning region
    const auto &objectInstance = navmesh.getObjectInstance(objectInstanceId);
    const auto transformedObjectResource = navmesh.getTransformedObjectResourceForRegion(objectInstanceId, objectInstance.regionId);

    // Build a map which holds edge links (to be used in constraints later)
    struct LinkedObjAndEdge {
      uint32_t objId;
      uint16_t edgeId;
    };
    std::map<int16_t, LinkedObjAndEdge> edgeLinkMap;
    for (const auto i : objectInstance.globalEdgeLinks) {
      if (i.edgeId != -1 && i.linkedObjGlobalId != -1) {
        // edge i.edgeId links to edge i.linkedObjEdgeId of object i.linkedObjGlobalId
        edgeLinkMap[i.edgeId].objId = i.linkedObjGlobalId;
        edgeLinkMap[i.edgeId].edgeId = i.linkedObjEdgeId;
      }
    }

    // Calculate region offset so we can trim it for our region
    const auto [originRegionX, originRegionY] = sro::position_math::sectorsFromWorldRegionId(objectInstance.regionId);
    const auto [ourRegionX, ourRegionY] = sro::position_math::sectorsFromWorldRegionId(region.id);
    const auto regionDx = ourRegionX-originRegionX;
    const auto regionDy = ourRegionY-originRegionY;

    for (const auto &cell : transformedObjectResource.cells) {
      // TODO: Is there ever an eventzone which is not the entire object?
      if (cell.eventZoneData) {
        // TODO: Not handling event zones yet
        // addObjEdgeWithTrim(transformedObjectResource.vertices.at(cell.vertex0),
        //                    transformedObjectResource.vertices.at(cell.vertex1),
        //                    inVertices,
        //                    inEdges);
        // addObjEdgeWithTrim(transformedObjectResource.vertices.at(cell.vertex1),
        //                    transformedObjectResource.vertices.at(cell.vertex2),
        //                    inVertices,
        //                    inEdges);
        // addObjEdgeWithTrim(transformedObjectResource.vertices.at(cell.vertex2),
        //                    transformedObjectResource.vertices.at(cell.vertex0),
        //                    inVertices,
        //                    inEdges);
      }
    }

    // Add all outline edges of the object
    for (int outlineEdgeIndex=0; outlineEdgeIndex<transformedObjectResource.outlineEdges.size(); ++outlineEdgeIndex) {
      const auto &edge = transformedObjectResource.outlineEdges.at(outlineEdgeIndex);

      // Determine flags for this edge
      std::vector<EdgeConstraintFlag> edgeFlags;
      if ((edge.flag & 1) || (edge.flag & 2)) {
        // Blocked
        if (edge.destCell != -1) {
          throw std::runtime_error("Expecting that the dest cell is always -1 since this is an outline edge");
        }
        edgeFlags.push_back(EdgeConstraintFlag::kGlobal);
        edgeFlags.push_back(EdgeConstraintFlag::kBlocking);
      } else if (edge.flag & 16) {
        // Bridge
        if (edge.destCell != -1) {
          throw std::runtime_error("Expecting that the dest cell is always -1 since this is an outline edge");
        }
        edgeFlags.push_back(EdgeConstraintFlag::kGlobal);
        edgeFlags.push_back(EdgeConstraintFlag::kBridge);
      } else if (edge.flag & 128) {
        // Siege
        // TODO: Not yet handling properly, calling it a global blocking edge for now
        if (edge.destCell != -1) {
          throw std::runtime_error("Expecting that the dest cell is always -1 since this is an outline edge");
        }
        edgeFlags.push_back(EdgeConstraintFlag::kGlobal);
        edgeFlags.push_back(EdgeConstraintFlag::kBlocking);
      } else if (edge.eventZoneData) {
        // Event zone
        // TODO: Not yet handling
        continue;
      } else {
        // Non-blocking edges are our way "onto" the object
        if (edge.destCell != -1) {
          throw std::runtime_error("Expecting that the dest cell is always -1 since this is an outline edge");
        }
        edgeFlags.push_back(EdgeConstraintFlag::kGlobal);
      }

      // Create constraint for edge
      ConstraintData constraint({objectInstanceId, transformedObjectResource.cellAreaIds[edge.srcCell]});
      for (const auto flag : edgeFlags) {
        constraint.edgeFlag |= flag;
      }

      // Check if the edge is part of a link
      const auto edgeLinkIt = edgeLinkMap.find(outlineEdgeIndex);
      if (edgeLinkIt != edgeLinkMap.end()) {
        const auto &linkedObjAndEdge = edgeLinkIt->second;
        ObjectLink link;
        link.srcObjectGlobalId = objectInstanceId;
        link.destObjectGlobalId = linkedObjAndEdge.objId;
        link.srcEdgeIndex = outlineEdgeIndex;
        link.destEdgeIndex = linkedObjAndEdge.edgeId;
        constraint.linkIdAndIsSource_ = createLinkAndGetId(link);
      }

      // Add the edge with the created constraint
      addObjEdgeWithTrim(transformedObjectResource.vertices.at(edge.srcVertex),
                         transformedObjectResource.vertices.at(edge.destVertex),
                         regionDx,
                         regionDy,
                         constraint,
                         inVertices,
                         inEdges);
    }

    // Add inline edges of the object
    for (const auto &edge : transformedObjectResource.inlineEdges) {
      if ((edge.flag & 1) || (edge.flag & 2)) {
        // Blocked
        if (edge.srcCell == -1) {
          throw std::runtime_error("Expecting that the src cell is not -1 since this is an inline edge");
        }
        if (edge.destCell == -1) {
          throw std::runtime_error("Expecting that the dest cell is not -1 since this is an inline edge");
        }
        if (transformedObjectResource.cellAreaIds[edge.srcCell] != transformedObjectResource.cellAreaIds[edge.destCell]) {
          throw std::runtime_error("Expecting that the cells on both sides of this edge are part of the same area");
        }
        ConstraintData constraint({objectInstanceId, transformedObjectResource.cellAreaIds[edge.srcCell]});
        constraint.edgeFlag |= EdgeConstraintFlag::kInternal;
        constraint.edgeFlag |= EdgeConstraintFlag::kBlocking;
        addObjEdgeWithTrim(transformedObjectResource.vertices.at(edge.srcVertex),
                           transformedObjectResource.vertices.at(edge.destVertex),
                           regionDx,
                           regionDy,
                           constraint,
                           inVertices,
                           inEdges);
      } else if (edge.flag & 128) {
        // Siege
        // TODO: Not yet handling
      } else if (edge.eventZoneData) {
        // Event zone
        // TODO: Not yet handling
      } else {
        // Non-blocking, don't really care about this edge
        // Internal bridge edges exist, we will just treat them as non-blocking
      }
    }
  }

  // ===============================================================================
  // =========================Ok, input data is transformed=========================
  // ===============================================================================

  // Some data init
	triangle::context *ctx;
	triangle::triangleio inputStruct;
	triangle::triangle_initialize_triangleio(&inputStruct);

	ctx = triangle::triangle_context_create();
  *(ctx->b) = pathfinder::BehaviorBuilder{}.getBehavior();

  // fill input structure vertices
  inputStruct.numberofpoints = static_cast<int>(inVertices.size());
  inputStruct.numberofpointattributes = 0;
	inputStruct.pointlist = (TRIANGLE_MACRO_REAL *) malloc((unsigned int) (2 * inVertices.size() * sizeof(TRIANGLE_MACRO_REAL)));
  int vertexIndex = 0;
  for (const auto &vertex : inVertices) {
    inputStruct.pointlist[2*vertexIndex] = vertex.x();
    inputStruct.pointlist[2*vertexIndex + 1] = vertex.y();
    ++vertexIndex;
  }

  // fill input structure edges
	inputStruct.numberofsegments = static_cast<int>(inEdges.size());
  inputStruct.segmentlist = (int *) malloc((unsigned int) (2 * inEdges.size() * sizeof(int)));
  inputStruct.segmentmarkerlist = (int *) malloc((unsigned int) (inEdges.size() * sizeof(int)));
  int edgeIndex = 0;
  for (const auto &edge : inEdges) {
    inputStruct.segmentlist[2*edgeIndex] = edge.vertex0;
    inputStruct.segmentlist[2*edgeIndex + 1] = edge.vertex1;
    inputStruct.segmentmarkerlist[edgeIndex] = edge.marker;
    ++edgeIndex;
  }

  // No "regions" or holes
  inputStruct.numberofregions = 0;
  inputStruct.numberofholes = 0;

  // generate mesh
  int meshCreateResult = triangle_mesh_create(ctx, &inputStruct);
  if (meshCreateResult < 0) {
    triangle::triangle_free_triangleio(&inputStruct);
    triangle::triangle_context_destroy(ctx);
    throw std::runtime_error("Error creating mesh "+std::to_string(meshCreateResult));
  }

  // Prepare data structures
  triangle::triangleio triangleData, triangleVoronoiData;
	triangle::triangle_initialize_triangleio(&triangleData);
	triangle::triangle_initialize_triangleio(&triangleVoronoiData);

  // Extract data from the context
  int copyResult = triangle_mesh_copy(ctx, &triangleData, 1, 1, &triangleVoronoiData);
  if (copyResult < 0) {
    triangle::triangle_free_triangleio(&triangleData);
    triangle::triangle_free_triangleio(&triangleVoronoiData);
    triangle::triangle_free_triangleio(&inputStruct);
    triangle::triangle_context_destroy(ctx);
    throw std::runtime_error("Error copying data");
  }

  SingleRegionNavmeshTriangulation navmeshTriangulation(navmesh, region, triangleData, triangleVoronoiData, std::move(constraintData), linkData_);

  // Post-processing
  markBlockedTerrainCells(navmeshTriangulation, region);

  // Cleanup
  triangle::triangle_free_triangleio(&triangleData);
  triangle::triangle_free_triangleio(&triangleVoronoiData);
  triangle::triangle_free_triangleio(&inputStruct);
  triangle::triangle_context_destroy(ctx);

  // Save the triangulation
  navmeshTriangulationMap_.emplace(region.id, std::move(navmeshTriangulation));
}

void NavmeshTriangulation::markBlockedTerrainCells(SingleRegionNavmeshTriangulation &navmeshTriangulation, const Region &region) const {
  // Mark cells(triangles) as blocked from the terrain perspective
  auto midpointOfTriangle = [](const auto &vertices) {
    const double sumX = std::get<0>(vertices).x() + std::get<1>(vertices).x() + std::get<2>(vertices).x();
    const double sumY = std::get<0>(vertices).y() + std::get<1>(vertices).y() + std::get<2>(vertices).y();
    return pathfinder::Vector{sumX/3.0, sumY/3.0};
  };
  auto pointToTilePos = [](const auto &point) {
    // return row,col
    return std::pair<int,int>{static_cast<int>(point.y()/20.0), static_cast<int>(point.x()/20.0)};
  };
  std::vector<bool> blockedTerrainTriangles(navmeshTriangulation.getTriangleCount(), false);
  for (std::size_t i=0; i<navmeshTriangulation.getTriangleCount(); ++i) {
    const auto triangleVertices = navmeshTriangulation.getTriangleVertices(static_cast<SingleRegionNavmeshTriangulation::IndexType>(i));
    const auto midpoint = midpointOfTriangle(triangleVertices);
    // This midpoint must land inside of some tile
    const auto tilePos = pointToTilePos(midpoint);
    if (region.enabledTiles[tilePos.first][tilePos.second] == false) {
      // Blocked tile, entire triangle must be blocked
      blockedTerrainTriangles[i] = true;
    }
  }

  navmeshTriangulation.setBlockedTerrainTriangles(std::move(blockedTerrainTriangles));
}

namespace geometry_helpers {

bool lineTrimToRegion(math::Vector3 &p1, math::Vector3 &p2, const double minX, const double minY, const double maxX, const double maxY) {
  constexpr const double kPrecisionTolerance = 1e-4;

  struct Boundary {
    double v1x, v1y;
    double v2x, v2y;
  };
  std::array<Boundary, 4> boundaries = {{{ minX, minY, minX, maxY},
                                         { minX, maxY, maxX, maxY},
                                         { maxX, maxY, maxX, minY},
                                         { maxX, minY, minX, minY}}};

  // Compare this line against all boundaries of the region
  for (const auto &boundary : boundaries) {
    // Check if lines intersect
    auto &x1 = p1.x;
    auto &y1 = p1.z;
    auto &x2 = p2.x;
    auto &y2 = p2.z;
    const double x3 = boundary.v1x;
    const double y3 = boundary.v1y;
    const double x4 = boundary.v2x;
    const double y4 = boundary.v2y;

    auto det = [](const double a, const double b, const double c, const double d) {
      return a*d-b*c;
    };

    double tNumerator = det(static_cast<double>(x1)-x3, static_cast<double>(x3)-x4, static_cast<double>(y1)-y3, static_cast<double>(y3)-y4);
    double uNumerator = det(static_cast<double>(x1)-x2, static_cast<double>(x1)-x3, static_cast<double>(y1)-y2, static_cast<double>(y1)-y3);
    double denom      = det(static_cast<double>(x1)-x2, static_cast<double>(x3)-x4, static_cast<double>(y1)-y2, static_cast<double>(y3)-y4);

    if (denom == 0) {
      continue;
    }

    const double t = tNumerator/denom;
    const double u = -uNumerator/denom;
    if (pathfinder::math::lessThan(0, t) &&
        !pathfinder::math::lessThan(1, t) &&
        pathfinder::math::lessThan(0, u) &&
        !pathfinder::math::lessThan(1, u)) {
      // Intersection lies on both segments
      // Trim the line based on the line it intersected with
      double intersectionX = x1+t*(static_cast<double>(x2)-x1);
      double intersectionY = y1+t*(static_cast<double>(y2)-y1);

      bool trimTheFirstPoint;
      if (pathfinder::math::equal(x3, x4, kPrecisionTolerance)) {
        // Vertical lines
        if (pathfinder::math::equal(y3, minY, kPrecisionTolerance)) {
          // Left side
          if (pathfinder::math::lessThan(x1, minX, kPrecisionTolerance)) {
            trimTheFirstPoint = true;
          } else {
            trimTheFirstPoint = false;
          }
        } else {
          // Right side
          if (pathfinder::math::lessThan(maxX, x1, kPrecisionTolerance)) {
            trimTheFirstPoint = true;
          } else {
            trimTheFirstPoint = false;
          }
        }
      } else {
        if (pathfinder::math::equal(x3, minX, kPrecisionTolerance)) {
          // Top side
          if (pathfinder::math::lessThan(maxY, y1, kPrecisionTolerance)) {
            trimTheFirstPoint = true;
          } else {
            trimTheFirstPoint = false;
          }
        } else {
          // Bottom side
          if (pathfinder::math::lessThan(y1, minY, kPrecisionTolerance)) {
            trimTheFirstPoint = true;
          } else {
            trimTheFirstPoint = false;
          }
        }
      }

      // Update line segment to exclude the part that isnt inside the region
      if (trimTheFirstPoint) {
        x1 = static_cast<float>(intersectionX);
        y1 = static_cast<float>(intersectionY);
        if (pathfinder::math::equal(x1, minX, kPrecisionTolerance)) {
          x1 = static_cast<float>(minX);
        }
        if (pathfinder::math::equal(y1, minY, kPrecisionTolerance)) {
          y1 = static_cast<float>(minY);
        }
        if (pathfinder::math::equal(x1, maxX, kPrecisionTolerance)) {
          x1 = static_cast<float>(maxX);
        }
        if (pathfinder::math::equal(y1, maxY, kPrecisionTolerance)) {
          y1 = static_cast<float>(maxY);
        }
      } else {
        x2 = static_cast<float>(intersectionX);
        y2 = static_cast<float>(intersectionY);
        if (pathfinder::math::equal(x2, minX, kPrecisionTolerance)) {
          x2 = static_cast<float>(minX);
        }
        if (pathfinder::math::equal(y2, minY, kPrecisionTolerance)) {
          y2 = static_cast<float>(minY);
        }
        if (pathfinder::math::equal(x2, maxX, kPrecisionTolerance)) {
          x2 = static_cast<float>(maxX);
        }
        if (pathfinder::math::equal(y2, maxY, kPrecisionTolerance)) {
          y2 = static_cast<float>(maxY);
        }
      }
    }
  }

  // If any point is outside of the region, then the entire line must be outside, return false
  bool pointOutside = false;
  pointOutside |= (pathfinder::math::lessThan(p1.x, minX, kPrecisionTolerance) || pathfinder::math::lessThan(maxX, p1.x, kPrecisionTolerance));
  pointOutside |= (pathfinder::math::lessThan(p1.z, minY, kPrecisionTolerance) || pathfinder::math::lessThan(maxY, p1.z, kPrecisionTolerance));
  pointOutside |= (pathfinder::math::lessThan(p2.x, minX, kPrecisionTolerance) || pathfinder::math::lessThan(maxX, p2.x, kPrecisionTolerance));
  pointOutside |= (pathfinder::math::lessThan(p2.z, minY, kPrecisionTolerance) || pathfinder::math::lessThan(maxY, p2.z, kPrecisionTolerance));
  return !pointOutside;
}

} // namespace geometry_helpers

} // namespace sro::navmesh::triangulation