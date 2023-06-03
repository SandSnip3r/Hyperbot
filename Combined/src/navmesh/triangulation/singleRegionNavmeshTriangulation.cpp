#include "math_helpers.h"
#include "singleRegionNavmeshTriangulation.hpp"

#include <functional>
#include <iostream>
#include <optional>
#include <queue>
#include <utility>

namespace navmesh {

namespace triangulation {

SingleRegionNavmeshTriangulation::SingleRegionNavmeshTriangulation(const navmesh::Navmesh &navmesh,
                                                                   const navmesh::Region &region,
                                                                   const triangle::triangleio &triangleData,
                                                                   const triangle::triangleio &triangleVoronoiData,
                                                                   std::vector<std::vector<ConstraintData>> &&constraintData,
                                                                   const std::vector<ObjectLink> &globalObjectLinks) :
      TriangleLibNavmesh(triangleData, triangleVoronoiData),
      navmesh_(navmesh),
      region_(region),
      constraintData_(std::move(constraintData)),
      globalObjectLinks_(globalObjectLinks),
      objectDatasForTriangles_(getTriangleCount()) {
  buildLinkData();
}

void SingleRegionNavmeshTriangulation::buildLinkData() {
  // Figure out which edges are a part of which links
  struct EdgesForLink {
    std::set<IndexType> edgesOnSrcSide, edgesOnDestSide;
  };
  std::map<LinkIdType, EdgesForLink> edgesForLinkMap;

  const auto edgeCount = getEdgeCount();
  for (IndexType edgeIndex=0; edgeIndex<edgeCount; ++edgeIndex) {
    const auto edgeMarker = getEdgeMarker(edgeIndex);
    if (edgeMarker >= 2) {
      const auto &edgeConstraints = getEdgeConstraintData(edgeMarker);
      for (const auto &constraint : edgeConstraints) {
        if (constraint.hasLink()) {
          auto &edges = edgesForLinkMap[constraint.getLinkId()];
          if (constraint.isOnSourceSideOfLink()) {
            edges.edgesOnSrcSide.insert(edgeIndex);
          } else {
            edges.edgesOnDestSide.insert(edgeIndex);
          }
        }
      }
    }
  }

  for (const auto &linkIdEdgesPair : edgesForLinkMap) {
    const auto linkId = linkIdEdgesPair.first;
    const auto &edgesForLink = linkIdEdgesPair.second;

    // Build list of edges and their vertices
    struct EdgeAndVertices {
      IndexType edgeIndex, v1Index, v2Index;
    };
    std::vector<EdgeAndVertices> srcSideEdges, destSideEdges;

    // Source side edges
    srcSideEdges.reserve(edgesForLink.edgesOnSrcSide.size());
    for (const auto edgeIndex : edgesForLink.edgesOnSrcSide) {
      const auto [v1Index, v2Index] = getEdgeVertexIndices(edgeIndex);
      srcSideEdges.push_back({edgeIndex, v1Index, v2Index});
    }

    // Destination side edges
    destSideEdges.reserve(edgesForLink.edgesOnDestSide.size());
    for (const auto edgeIndex : edgesForLink.edgesOnDestSide) {
      const auto [v1Index, v2Index] = getEdgeVertexIndices(edgeIndex);
      destSideEdges.push_back({edgeIndex, v1Index, v2Index});
    }

    // Lambda for comparison of two vertices
    const auto vertexIsGreater = [](const auto &v1, const auto &v2) {
      if (v1.x() == v2.x()) {
        if (v1.y() > v2.y()) {
          return true;
        }
      } else if (v1.x() > v2.x()) {
        return true;
      }
      return false;
    };

    // Sort vertices within edges
    for (auto *edgeList : {&srcSideEdges, &destSideEdges}) {
      for (auto &edge : *edgeList) {
        const auto &v1 = getVertex(edge.v1Index);
        const auto &v2 = getVertex(edge.v2Index);
        if (vertexIsGreater(v1, v2)) {
          std::swap(edge.v1Index, edge.v2Index);
        }
      }
    }

    // Get the first and last vertices of each edge
    const auto getIndicesOfFirstAndLastVertexOfEdge = [this, &vertexIsGreater](const auto &edgeList) {
      auto edgeFirstVertexIndex = edgeList.front().v1Index;
      auto edgeLastVertexIndex = edgeFirstVertexIndex;
      auto edgeFirstVertex = getVertex(edgeFirstVertexIndex);
      auto edgeLastVertex = edgeFirstVertex;
      for (const auto &edge : edgeList) {
        const auto &v1 = getVertex(edge.v1Index);
        const auto &v2 = getVertex(edge.v2Index);
        if (vertexIsGreater(v2, edgeLastVertex)) {
          edgeLastVertex = v2;
          edgeLastVertexIndex = edge.v2Index;
        }
        if (vertexIsGreater(edgeFirstVertex, v1)) {
          edgeFirstVertex = v1;
          edgeFirstVertexIndex = edge.v1Index;
        }
      }
      return std::make_pair(edgeFirstVertexIndex, edgeLastVertexIndex);
    };
    const auto [srcEdgeFirstVertexIndex, srcEdgeLastVertexIndex] = getIndicesOfFirstAndLastVertexOfEdge(srcSideEdges);
    const auto [destEdgeFirstVertexIndex, destEdgeLastVertexIndex] = getIndicesOfFirstAndLastVertexOfEdge(destSideEdges);
    const auto &srcEdgeFirstVertex = getVertex(srcEdgeFirstVertexIndex);
    const auto &srcEdgeLastVertex = getVertex(srcEdgeLastVertexIndex);
    const auto &destEdgeFirstVertex = getVertex(destEdgeFirstVertexIndex);
    const auto &destEdgeLastVertex = getVertex(destEdgeLastVertexIndex);

    {
      struct LinkTriangle {
        pathfinder::Vector v1, v2, v3;
      };

      // Triangulate the link area
      const auto triangulatedLinkArea = [&srcEdgeFirstVertex, &srcEdgeLastVertex, &destEdgeFirstVertex, &destEdgeLastVertex]() -> std::vector<LinkTriangle> {
        pathfinder::Vector pointOfIntersection;
        const auto intersectionResult = pathfinder::math::intersect(srcEdgeFirstVertex, srcEdgeLastVertex, destEdgeFirstVertex, destEdgeLastVertex, &pointOfIntersection);
        if (intersectionResult == pathfinder::math::IntersectionResult::kNone) {
          // These two edges do not intersect, create two triangles which make up this quadrilateral
          return {LinkTriangle{srcEdgeFirstVertex, destEdgeFirstVertex, srcEdgeLastVertex},
                  LinkTriangle{destEdgeFirstVertex, srcEdgeLastVertex, destEdgeLastVertex}};
        } else if (intersectionResult == pathfinder::math::IntersectionResult::kInfinite) {
          throw std::runtime_error("Two link edges have infinite overlap. Shouldn't be possible");
        }
        // Must be a single point of overlap
        if (srcEdgeFirstVertex == destEdgeFirstVertex) {
          // Share starting endpoints
          return {LinkTriangle{srcEdgeFirstVertex, srcEdgeLastVertex, destEdgeLastVertex}};
        } else if (srcEdgeLastVertex == destEdgeLastVertex) {
          // Share ending endpoints
          return {LinkTriangle{srcEdgeFirstVertex, destEdgeFirstVertex, srcEdgeLastVertex}};
        } else if (srcEdgeFirstVertex == destEdgeLastVertex ||
                  srcEdgeLastVertex == destEdgeFirstVertex) {
          // Share opposite endpoints
          throw std::runtime_error("Im not sure what the implications of this case are");
        } else {
          // The single point of overlap is mid-line for at least one of the edges
          if (pathfinder::math::equal(pointOfIntersection, srcEdgeFirstVertex) ||
              pathfinder::math::equal(pointOfIntersection, srcEdgeLastVertex) ||
              pathfinder::math::equal(pointOfIntersection, destEdgeFirstVertex) ||
              pathfinder::math::equal(pointOfIntersection, destEdgeLastVertex)) {
            // The point of overlap is at one of the lines endpoints
            // TODO: Handle
            // This is pretty improbable
            throw std::runtime_error("The point of overlap is at one of the lines endpoints");
          }
          // This overlap is mid-line for both edges, create two triangles
          return {LinkTriangle{srcEdgeFirstVertex, pointOfIntersection, destEdgeFirstVertex},
                  LinkTriangle{srcEdgeLastVertex, pointOfIntersection, destEdgeLastVertex}};
        }
      }();

      const auto triangleOverlapsWithLink = [this, &triangulatedLinkArea](const auto triangleIndex) {
        const auto [vertex1, vertex2, vertex3] = getTriangleVertices(triangleIndex);
        int triInLinkIdx=0;
        for (const auto &triangleInLink : triangulatedLinkArea) {
          constexpr const double kEpsilon = 0.3;
          // 0.27 is big enough to rule out some errors
          // However, there are some tiny triangles which should be included but are not because their area is ~0.06
          // TODO: Figure out how to include small triangles as well
          //  (Big TODO)
          if (pathfinder::math::trianglesOverlap(vertex1, vertex2, vertex3, triangleInLink.v1, triangleInLink.v2, triangleInLink.v3, false, kEpsilon)) {
            return true;
          }
          ++triInLinkIdx;
        }
        return false;
      };

      // Loop over all triangles, if that triangle overlaps with a triangle of the link, add it
      for (int triangleIndex=0; triangleIndex<getTriangleCount(); ++triangleIndex) {
        if (triangleOverlapsWithLink(triangleIndex)) {
          linkDataMap_[linkId].accessibleTriangleIndices.insert(triangleIndex);
        }
      }
    }
  }
}

std::optional<SingleRegionNavmeshTriangulation::LinkIdType> SingleRegionNavmeshTriangulation::getLinkIdForTriangle(const IndexType triangleIndex) const {
  for (const auto &link : linkDataMap_) {
    if (link.second.accessibleTriangleIndices.find(triangleIndex) != link.second.accessibleTriangleIndices.end()) {
      return link.first;
    }
  }
  return {};
}

const std::vector<ConstraintData>& SingleRegionNavmeshTriangulation::getEdgeConstraintData(const MarkerType edgeMarker) const {
  if (edgeMarker < 2) {
    throw std::invalid_argument("Asking for constraint data for a non-user-defined edge marker");
  }
  if (edgeMarker-2 >= constraintData_.size()) {
    throw std::invalid_argument("This marker references data which does not exist");
  }
  return constraintData_[edgeMarker-2];
}

void SingleRegionNavmeshTriangulation::setBlockedTerrainTriangles(std::vector<bool> &&blockedTerrainTriangles) {
  blockedTerrainTriangles_ = blockedTerrainTriangles;
}

bool SingleRegionNavmeshTriangulation::terrainIsBlockedUnderTriangle(const IndexType triangleIndex) const {
  if (triangleIndex >= getTriangleCount()) {
    throw std::runtime_error("Trying to check if terrain is blocked for triangle which does not exist");
  }
  return blockedTerrainTriangles_[triangleIndex];
}

void SingleRegionNavmeshTriangulation::addObjectDataForTriangle(const IndexType triangleIndex, const ObjectData &objectData) {
  if (triangleIndex >= getTriangleCount()) {
    throw std::runtime_error("Trying to add object data for triangle which does not exist");
  }
  objectDatasForTriangles_[triangleIndex].push_back(objectData);
}

const std::vector<ObjectData>& SingleRegionNavmeshTriangulation::getObjectDatasForTriangle(const IndexType triangleIndex) const {
  if (triangleIndex >= getTriangleCount()) {
    throw std::runtime_error("Trying to get object instances for triangle which does not exist");
  }
  return objectDatasForTriangles_[triangleIndex];
}

bool SingleRegionNavmeshTriangulation::agentFitsThroughEdge(const IndexType edgeIndex, const double agentRadius) const {
  const auto [edgeVertex1, edgeVertex2] = getEdge(edgeIndex);
  return !pathfinder::math::lessThan(pathfinder::math::distance(edgeVertex1, edgeVertex2), (agentRadius*2));
  // TODO: Lots of work to do here
  //  The vertices of the edge could be unconstrained and thus we could pass through them
}

std::vector<SingleRegionNavmeshTriangulation::State> SingleRegionNavmeshTriangulation::getSuccessors(const State &currentState, const std::optional<State> goalState, const double agentRadius) const {
  // A state corresponds to a single triangle triangle (but a triangle can have multiple states)
  // This triangle has 3 edges (obviously)
  // 1 of these edges is the edge that we came though (unless we're starting in this triangle, then there isnt an entry edge index)
  //  I'm fairly confident, but not 100% sure, that we will never want to return through the edge which we entered this triangle

  auto agentFitsThroughTriangle = [this, &agentRadius](const IndexType triangleIndex, const IndexType entryEdgeIndex, const IndexType exitEdgeIndex) -> bool {
    // TODO: LOTS of work to do here
    return true;
  };

  const auto getSuccessorStateThroughEdgeIfPossible = [this, &currentState, agentRadius](const IndexType entryEdgeIndex, const IndexType neighborTriangleIndex) -> std::optional<State> {
    if (currentState.hasEntryEdgeIndex() && entryEdgeIndex == currentState.getEntryEdgeIndex()) {
      // Don't return through the edge which we entered through
      // TODO: In the takla bridge case, it would make sense to return through the edge which we entered, only if the resulting state is different than the previous state
      // TODO: To solve this, maybe we need to have access to our previous state too?
      //  That info might be hard to get
      // TODO: Maybe we just dont do this check, and let the using algorithm filter out already-seen states
      return {};
    }

    if (!agentFitsThroughEdge(entryEdgeIndex, agentRadius)) {
      // Agent cannot fit through this edge, no successor this way
      return {};
    }

    const auto edgeMarker = getEdgeMarker(entryEdgeIndex);

    const auto exitingLink = [this, &edgeMarker, &currentState]{
      if (!currentState.isTraversingLink()) {
        throw std::runtime_error("Checking if we're exiting a link, but we're not in a link to start with");
      }
      const auto &edgeConstraintData = getEdgeConstraintData(edgeMarker);
      for (const auto &constraint : edgeConstraintData) {
        if (constraint.hasLink() && currentState.getLinkId() == constraint.getLinkId()) {
          return true;
        }
      }
      return false;
    };

    if (edgeMarker == 0) {
      // Unconstrained edge (no data, must be a result of triangulation); can always pass through
      // If on terrain, we are still on the terrain
      // If on an object, we are still on that same object
      // Create a state similar to our existing state, only update triangle and entry edge
      auto newState = currentState;
      newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
      return newState;
    } else if (edgeMarker == 1) {
      // Constrained edge as computed by the triangulation library
      //  We ensure that these edge cannot exist (by making sure the convex hull exists)
      throw std::runtime_error("Constrained edges with marker 1 should not exist");
    } else if (currentState.isTraversingLink()) {
      // We are currently inside of a link
      // As long as we stay within the link, we can cross any edge
      if (exitingLink()) {
        // We are leaving the link
        // Create state which is now outside the link and on the new object
        auto newState = currentState;
        newState.resetLinkId();
        newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);

        // Figure out which object to put this state on
        const auto &thisLink = globalObjectLinks_.at(currentState.getLinkId());
        const auto objectDatasForTriangle = getObjectDatasForTriangle(neighborTriangleIndex);
        using ObjectIdType = decltype(ObjectData::objectInstanceId);
        ObjectIdType objId;
        bool src{false}, dest{false};
        for (const auto &objectDataForTriangle : objectDatasForTriangle) {
          if (thisLink.srcObjectGlobalId == objectDataForTriangle.objectInstanceId) {
            src = true;
            objId = objectDataForTriangle.objectInstanceId;
          } else if (thisLink.destObjectGlobalId == objectDataForTriangle.objectInstanceId) {
            dest = true;
            objId = objectDataForTriangle.objectInstanceId;
          }
        }
        if (src && dest) {
          throw std::runtime_error("Thought it was impossible for both objects to be on this triangle");
        }
        // TODO: Get the proper object area
        // Until we have that, check if there even is multiple object areas for this triangle
        using ObjectAreaType = decltype(ObjectData::objectAreaId);
        std::set<ObjectAreaType> objectAreaSet;
        for (const auto &objectDataForTriangle : objectDatasForTriangle) {
          if (objectDataForTriangle.objectInstanceId == objId) {
            objectAreaSet.insert(objectDataForTriangle.objectAreaId);
          }
        }
        if (objectAreaSet.size() > 1) {
          throw std::runtime_error("There are multiple areas for this triangle & object");
        }
        newState.setObjectData({objId, *objectAreaSet.begin()});
        return newState;
      } else {
        // Still somewhere between the two linked edges
        // We can avoid any constraints, we just must stay within the acceptable area
        auto acceptableTrianglesIt = linkDataMap_.find(currentState.getLinkId());
        if (acceptableTrianglesIt == linkDataMap_.end()) {
          throw std::runtime_error("We have no list of triangles for this link");
        }
        if (acceptableTrianglesIt->second.accessibleTriangleIndices.find(neighborTriangleIndex) == acceptableTrianglesIt->second.accessibleTriangleIndices.end()) {
          // This is not a valid triangle to continue onto; no successor
          return {};
        }

        // Valid triangle within the link, generate a successor on this triangle
        auto newState = currentState;
        newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
        return newState;
      }
    } else {
      // We are currently not traversing a link
      // Edge is some kind of constraint (might not necessarily be blocking)
      const auto &edgeConstraintData = getEdgeConstraintData(edgeMarker);
      std::optional<State> potentialSuccessorState;
      for (const auto &constraint : edgeConstraintData) {
        if (constraint.forTerrain() && constraint.is(EdgeConstraintFlag::kGlobal)) {
          // Global edge of the region; we do not leave the region within this context
          return {};
        } else {
          if (currentState.isOnObject()) {
            // We are currently on some object
            if (constraint.forObject()) {
              // This edge is a constraint of an object (and not a constraint of the terrain, again, might not be blocking)
              if (constraint.hasLink()) {
                // This edge is a link edge. Maybe we are entering a link
                const auto linkId = constraint.getLinkId();
                if (linkDataMap_.find(linkId) != linkDataMap_.end()) {
                  const auto &link = globalObjectLinks_.at(linkId);
                  if (link.srcObjectGlobalId == currentState.getObjectData().objectInstanceId ||
                      link.destObjectGlobalId == currentState.getObjectData().objectInstanceId) {
                    const auto &linkData = linkDataMap_.at(linkId);
                    if (linkData.accessibleTriangleIndices.find(neighborTriangleIndex) != linkData.accessibleTriangleIndices.end()) {
                      // This link is for our object
                      State newState(neighborTriangleIndex, entryEdgeIndex);
                      newState.setLinkId(linkId);
                      return newState;
                    } else {
                      // This link is for our object but we're not entering the link
                      State newState{currentState};
                      newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
                      return newState;
                    }
                  }
                } else {
                  // Link data does not actually exist
                  //  This is likely a result of triangle overlap calculation issues
                  //  TODO: Delete this check and block once the triangle overlap calculation is fixed
                  // For now, fall into the case below
                  // The link wont work correctly if the objects overlap
                }
              }
              // Fact: We are not entering nor inside a link
              if (currentState.getObjectData() == constraint.getObjectData()) {
                // This edge is a constraint of the object that we are on and the same area in that object that we're on
                if (constraint.is(EdgeConstraintFlag::kBlocking)) {
                  // Blocking edge for our object, cannot cross; no successor
                  return {};
                } else if (constraint.is(EdgeConstraintFlag::kGlobal)) {
                  // External edge of the object
                  if (constraint.is(EdgeConstraintFlag::kBridge)) {
                    // From within the object, external bridge edges are blocking; no successor
                    return {};
                  } else {
                    // This edge is non-blocking and is an outline edge of the object
                    //  Crossing this edge will lead us off the object

                    // Check if this constraint is a link to an object that the neighbor triangle is in
                    std::optional<ObjectData> goalObjectData;
                    // if (constraint.hasLink()) {
                    //   const auto &objectDataForNeighborTriangle = getObjectDatasForTriangle(neighborTriangleIndex);
                    //   const auto linkId = constraint.getLinkId();
                    //   const auto &link = globalObjectLinks_.at(linkId);
                    //   for (const auto &od : objectDataForNeighborTriangle) {
                    //     if (od.objectInstanceId == link.destObjectGlobalId) {
                    //       // TODO
                    //       if (goalObjectData) {
                    //         std::cout << "Whoa! Multiple matching destinations for the link" << std::endl;
                    //       }
                    //       goalObjectData = od;
                    //     }
                    //   }
                    // }
                    if (!goalObjectData.has_value() && blockedTerrainTriangles_[neighborTriangleIndex]) {
                      // Trying to leave the object onto blocked terrain; no successor
                      return {};
                    }

                    // Leaving the object. Either onto terrain, or onto linked object
                    // Create a state which is on the terrain or on the linked object, in the new triangle, and with the entry edge
                    auto newState = currentState;
                    newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
                    if (goalObjectData) {
                      newState.setObjectData(*goalObjectData);
                    } else {
                      newState.setOnTerrain();
                    }
                    return newState;
                  }
                } else {
                  // Edge is not blocked and is an internal edge of our current object and current area (and is a constraint edge)
                  //  These shouldn't even exit (because it shouldnt be a constraint edge)
                  throw std::runtime_error("Edge is not blocked and is an internal edge of our current object (and is a constraint edge)");
                }
              } else {
                // This edge is a constraint of some other object or a different area of our current object
                //  Collision is not checked against objects other than the one we're on or other areas
                auto newState = currentState;
                // Create a state which is still on this same object, in the new triangle, and with the entry edge
                newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
                // We will only return this state if there is no other constraint which blocks us
                potentialSuccessorState = newState;
              }
            } else {
              // We are on an object and this constrained edge is not a constraint of any object (must be a terrain constraint)
              // Must be an internal edge of the terrain
              //  We can pass through this edge

              State newState = currentState;
              // Create a state which is still on this same object, in the new triangle, and with the entry edge
              newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
              return newState;
            }
          } else {
            // We are on the terrain
            if (constraint.forObject()) {
              // This edge is a constraint of an object (and not a constraint of the terrain, again, might not be blocking)
              // This edge is a constraint of the object that we are on
              if (constraint.is(EdgeConstraintFlag::kGlobal)) {
                // Only check collision against external edges of objects
                if (constraint.is(EdgeConstraintFlag::kBlocking)) {
                  // Cannot pass through; no successor
                  return {};
                } else if (constraint.is(EdgeConstraintFlag::kBridge)) {
                  // Can "pass through", we stay on terrain because we're going under some kind of bridge
                  // Create a state which is (still) on the terrain, in the new triangle, and with the entry edge
                  auto newState = currentState;
                  newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
                  return newState;
                } else {
                  // Unblocked external edge of object
                  bool weAreUnderTheObject{false};
                  const auto &objectDatasForThisTriangle = getObjectDatasForTriangle(currentState.getTriangleIndex());
                  if (std::find_if(objectDatasForThisTriangle.begin(), objectDatasForThisTriangle.end(), [&constraint](const auto &objectData){
                    return (objectData.objectInstanceId == constraint.getObjectData().objectInstanceId);
                  }) != objectDatasForThisTriangle.end()) {
                    // We are on the terrain, but this triangle also overlaps with the same object as is referenced by this edge
                    //  We must be underneath the object
                    const auto &objectDatasForNeighborTriangle = getObjectDatasForTriangle(neighborTriangleIndex);
                    if (std::find_if(objectDatasForNeighborTriangle.begin(), objectDatasForNeighborTriangle.end(), [&constraint](const auto &objectData){
                      return (objectData.objectInstanceId == constraint.getObjectData().objectInstanceId);
                    }) == objectDatasForNeighborTriangle.end()) {
                      // The neighbor triangle does not overlap with this object, we must be coming out from underneath the object
                      weAreUnderTheObject = true;
                    }
                  }
                  if (weAreUnderTheObject) {
                    // Create a state which is still on the terrain, in the new triangle, and with the entry edge
                    auto newState = currentState;
                    newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
                    return newState;
                  } else {
                    // Create a state which is now on this object, in the new triangle, and with the entry edge
                    auto newState = currentState;
                    newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
                    newState.setObjectData(constraint.getObjectData());
                    return newState;
                  }
                }
              } else {
                // On the terrain, internal edge of an object, I think collision is not checked in this case, even if blocking
                //  Can pass through
                // Create a state which is (still) on the terrain, in the new triangle, and with the entry edge
                auto newState = currentState;
                newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
                return newState;
              }
            } else {
              // Must be constrained by the terrain
              //  Global edges are already handled, so it must be an internal edge
              if (!constraint.is(EdgeConstraintFlag::kBlocking)) {
                // Can pass through
                // Create a state which is (still) on the terrain, in the new triangle, and with the entry edge
                auto newState = currentState;
                newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
                return newState;
              } else {
                // Cannot pass through; no successor
                return {};
              }
            }
          }
        }
      }
      if (potentialSuccessorState.has_value()) {
        return *potentialSuccessorState;
      }
    }
    throw std::runtime_error("Unhandled case for successor");
  };
  
  if (currentState.isGoal()) {
    throw std::runtime_error("Trying to get successors of goal");
  }

  const auto triangleIndexForState = currentState.getTriangleIndex();

  if (goalState.has_value()) {
    if (currentState.isSameTriangleAs(*goalState)) {
      // This is the goal, only successor is the goal point itself
      auto newGoalState{currentState};
      newGoalState.setIsGoal(true);
      return {newGoalState};
    }
  }

  if (triangleIndexForState >= getTriangleCount()) {
    throw std::runtime_error("Triangle is not in data");
  }
  
  std::vector<State> successors;
  // For each neighboring triangle
  const auto &[neighborAcrossEdge1, neighborAcrossEdge2, neighborAcrossEdge3] = getTriangleNeighborsWithSharedEdges(triangleIndexForState);
  for (const auto &neighborAcrossEdge : {neighborAcrossEdge1, neighborAcrossEdge2, neighborAcrossEdge3}) {
    if (neighborAcrossEdge) {
      // Neighbor exists
      auto possibleSuccessor = getSuccessorStateThroughEdgeIfPossible(neighborAcrossEdge->sharedEdgeIndex, neighborAcrossEdge->neighborTriangleIndex);
      if (possibleSuccessor) {
        successors.push_back(*possibleSuccessor);
      }
    }
  }
  return successors;
}

std::vector<SingleRegionNavmeshTriangulation::State> SingleRegionNavmeshTriangulation::getNeighborsInObjectArea(const State &currentState) const {
  const auto triangleIndexForState = currentState.getTriangleIndex();

  if (triangleIndexForState >= getTriangleCount()) {
    throw std::runtime_error("Triangle is not in data");
  }

  if (!currentState.isOnObject()) {
    throw std::runtime_error("Expecting a state that is on an object");
  }

  auto getSuccessorStateThroughEdgeIfPossible = [this, &currentState](const IndexType entryEdgeIndex, const IndexType neighborTriangleIndex) -> std::optional<State> {
    if (currentState.hasEntryEdgeIndex() && entryEdgeIndex == currentState.getEntryEdgeIndex()) {
      // Don't return through the edge which we entered through
      return {};
    }

    const auto edgeMarker = getEdgeMarker(entryEdgeIndex);
    if (edgeMarker == 0) {
      // Unconstrained edge (no data, must be a result of triangulation); can always pass through
      // If on terrain, we are still on the terrain
      // If on an object, we are still on that same object
      // Create a state similar to our existing state, only update triangle and entry edge
      State newState = currentState;
      newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
      return newState;
    } else if (edgeMarker == 1) {
      // Constrained edge as computed by the triangulation library
      //  It will be possible to encounter these, but im not sure how to handle them
      // For now, we expect that this is only encountered at the outline of the region, which we are not handling yet
      // TODO!
      //  In fact, I think we should ensure that these edge cannot exist (by making sure the convex hull exists)
      throw std::runtime_error("Not yet handling constrained edges with marker 1");
    } else {
      // Edge is some kind of constraint (might not necessarily be blocking)
      if (currentState.isOnObject()) {
        // We are currently on some object
        const auto edgeConstraintData = getEdgeConstraintData(edgeMarker);
        std::optional<State> potentialSuccessorState;
        for (const auto &constraint : edgeConstraintData) {
          if (constraint.forObject()) {
            // This edge is a constraint of an object (and not a constraint of the terrain, again, might not be blocking)
            if (currentState.getObjectData() == constraint.getObjectData()) {
              // This edge is a constraint of the object that we are on and the same area in that object that we're on
              if (constraint.is(EdgeConstraintFlag::kGlobal)) {
                // External edge of object; cannot leave object
                return {};
              } else if (constraint.is(EdgeConstraintFlag::kInternal)) {
                // Edge an internal edge of our current object and current area (and is a constraint edge) and we're ignoring whether it's blocked or not
                State newState = currentState;
                // Create a state which is still on this same object, in the new triangle, and with the entry edge
                newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
                return newState;
              } else {
                throw std::runtime_error("Edge is neither global nor internal");
              }
            } else {
              // This edge is a constraint of some other object or a different area of our current object
              //  Collision is not checked against objects other than the one we're on or other areas
              State newState = currentState;
              // Create a state which is still on this same object, in the new triangle, and with the entry edge
              newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
              // We will only return this state if there is no other constraint which blocks us
              potentialSuccessorState = newState;
            }
          } else {
            // We are on an object and this constrained edge is not a constraint of any object (must be a terrain constraint)
            // Must be an internal edge of the terrain
            //  We can pass through this edge

            State newState = currentState;
            // Create a state which is still on this same object, in the new triangle, and with the entry edge
            newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
            return newState;
          }
        }
        if (potentialSuccessorState.has_value()) {
          return *potentialSuccessorState;
        }
      }
    }
    throw std::runtime_error("Unhandled case for successor");
  };

  std::vector<State> successors;
  // For each neighboring triangle
  const auto &[neighborAcrossEdge1, neighborAcrossEdge2, neighborAcrossEdge3] = getTriangleNeighborsWithSharedEdges(triangleIndexForState);
  for (const auto &neighborAcrossEdge : {neighborAcrossEdge1, neighborAcrossEdge2, neighborAcrossEdge3}) {
    if (neighborAcrossEdge) {
      // Neighbor exists
      auto possibleSuccessor = getSuccessorStateThroughEdgeIfPossible(neighborAcrossEdge->sharedEdgeIndex, neighborAcrossEdge->neighborTriangleIndex);
      if (possibleSuccessor) {
        successors.push_back(*possibleSuccessor);
      }
    }
  }
  return successors;
}

pathfinder::Vector SingleRegionNavmeshTriangulation::to2dPoint(const sro::math::Vector3 &point) {
  // Convert our 3d point into the pathfinder's 2d point type
  return {point.x, point.z};
}

SingleRegionNavmeshTriangulation::State SingleRegionNavmeshTriangulation::createStateForPoint(const sro::math::Vector3 &point, const IndexType triangleIndex) const {
  State result{triangleIndex};
  const auto &objectDatas = getObjectDatasForTriangle(triangleIndex);

  // First, find the object which is closest to our y-value
  float minHeightDifference = std::numeric_limits<float>::max();
  std::optional<ObjectData> closestObjectData;
  for (const auto &objectData : objectDatas) {
    const auto &objectInstance = navmesh_.getObjectInstance(objectData.objectInstanceId);
    const auto &objectResource = navmesh_.getObjectResource(objectInstance.objectId);
    const auto transformedPoint = navmesh_.transformPointIntoObjectFrame(point, region_.id, objectData.objectInstanceId);
    const auto heightOnObject = objectResource.getHeight(transformedPoint, objectData.objectAreaId) + objectInstance.center.y;
    const auto heightDiff = std::abs(heightOnObject-point.y);
    if (heightDiff < minHeightDifference) {
      minHeightDifference = heightDiff;
      closestObjectData = objectData;
    }
  }

  // Next, check if the terrain is closer to the y-value (if the terrain even is valid at this point)
  bool terrainIsCloser{false};
  if (!blockedTerrainTriangles_[triangleIndex]) {
    const float heightOnTerrain = region_.getHeightAtPoint({static_cast<float>(point.x), 0.0f, static_cast<float>(point.z)});
    const auto heightDiff = std::abs(heightOnTerrain-point.y);
    if (heightDiff < minHeightDifference) {
      terrainIsCloser = true;
    }
  }

  if (terrainIsCloser) {
    // Create state for terrain
    result.setOnTerrain();
  } else {
    if (closestObjectData.has_value()) {
      // Create state for object
      result.setObjectData(*closestObjectData);
    } else {
      // There is no terrain nor object at this point
      throw std::runtime_error("Asking for state of invalid point");
    }
  }

  return result;
}

SingleRegionNavmeshTriangulation::State SingleRegionNavmeshTriangulation::createStartState(const sro::math::Vector3 &point, const IndexType triangleIndex) const {
  return createStateForPoint(point, triangleIndex);
}

SingleRegionNavmeshTriangulation::State SingleRegionNavmeshTriangulation::createGoalState(const sro::math::Vector3 &point, const IndexType triangleIndex) const {
  auto state = createStateForPoint(point, triangleIndex);
  state.setIsGoal(true);
  return state;
}

std::ostream& operator<<(std::ostream &stream, const ConstraintData &data) {
  stream << "((";
  if (data.objectData_) {
    stream << data.objectData_->objectInstanceId << ',' << data.objectData_->objectAreaId;
  } else {
    stream << "<no object>";
  }
  stream << "),";
  if (data.edgeFlag == EdgeConstraintFlag::kNone) {
    stream << "None";
  } else {
    bool one{false};
    if (data.is(EdgeConstraintFlag::kInternal)) {
      stream << "Internal";
      one = true;
    }
    if (data.is(EdgeConstraintFlag::kGlobal)) {
      if (one) {
        stream << ',';
      }
      stream << "Global";
      one = true;
    }
    if (data.is(EdgeConstraintFlag::kBlocking)) {
      if (one) {
        stream << ',';
      }
      stream << "Blocking";
      one = true;
    }
    if (data.is(EdgeConstraintFlag::kBridge)) {
      if (one) {
        stream << ',';
      }
      stream << "Bridge";
    }
  }
  if (data.linkIdAndIsSource_) {
    stream << ',' << data.linkIdAndIsSource_->first << ',' << (data.linkIdAndIsSource_->second ? 'S' : 'D');
  }
  stream << ')';
  return stream;
}

bool operator==(const ConstraintData &a, const ConstraintData &b) {
  return ((a.objectData_ == b.objectData_) && (a.edgeFlag == b.edgeFlag) && (a.linkIdAndIsSource_ == b.linkIdAndIsSource_));
}

bool operator==(const ObjectLink &a, const ObjectLink &b) {
  return ((a.srcObjectGlobalId == b.srcObjectGlobalId) &&
          (a.destObjectGlobalId == b.destObjectGlobalId) &&
          (a.srcEdgeIndex == b.srcEdgeIndex) &&
          (a.destEdgeIndex == b.destEdgeIndex));
}

EdgeConstraintFlag operator&(const EdgeConstraintFlag a, const EdgeConstraintFlag b) {
  return static_cast<EdgeConstraintFlag>(static_cast<std::underlying_type<EdgeConstraintFlag>::type>(a) &
                                         static_cast<std::underlying_type<EdgeConstraintFlag>::type>(b));
}

EdgeConstraintFlag operator|(const EdgeConstraintFlag a, const EdgeConstraintFlag b) {
  return static_cast<EdgeConstraintFlag>(static_cast<std::underlying_type<EdgeConstraintFlag>::type>(a) |
                                         static_cast<std::underlying_type<EdgeConstraintFlag>::type>(b));
}

EdgeConstraintFlag& operator|=(EdgeConstraintFlag &a, const EdgeConstraintFlag b) {
  return (a = (a|b));
}

ConstraintData::ConstraintData(const ObjectData &objectData) : objectData_(objectData) {}

bool ConstraintData::is(const EdgeConstraintFlag flag) const {
  return (edgeFlag & flag) != EdgeConstraintFlag::kNone;
}

bool ConstraintData::forTerrain() const {
  return !forObject();
}

bool ConstraintData::forObject() const {
  return objectData_.has_value();
}

const ObjectData& ConstraintData::getObjectData() const {
  if (!objectData_) {
    throw std::runtime_error("Trying to get object data for a constraint that isnt for an object");
  }
  return objectData_.value();
}

bool ConstraintData::hasLink() const {
  return linkIdAndIsSource_.has_value();
}

uint32_t ConstraintData::getLinkId() const {
  if (!linkIdAndIsSource_.has_value()) {
    throw std::runtime_error("Trying to get non-existent link for constraint");
  }
  return linkIdAndIsSource_->first;
}

bool ConstraintData::isOnSourceSideOfLink() const {
  if (!linkIdAndIsSource_.has_value()) {
    throw std::runtime_error("Trying to get non-existent link for constraint");
  }
  return linkIdAndIsSource_->second;
}


} // namespace triangulation

} // namespace navmesh
