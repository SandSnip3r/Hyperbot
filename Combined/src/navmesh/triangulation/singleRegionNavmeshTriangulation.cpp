#include "math_helpers.h"
#include "singleRegionNavmeshTriangulation.hpp"

#include <iostream>
#include <optional>

namespace navmesh {

namespace triangulation {

SingleRegionNavmeshTriangulation::SingleRegionNavmeshTriangulation(const navmesh::Navmesh &navmesh, const navmesh::Region &region, const triangle::triangleio &triangleData, const triangle::triangleio &triangleVoronoiData, std::vector<ConstraintData> &&constraintData) : TriangleLibNavmesh(triangleData, triangleVoronoiData), navmesh_(navmesh), region_(region), constraintData_(std::move(constraintData)), objectDatasForTriangles_(getTriangleCount()) {
  //
}

const ConstraintData& SingleRegionNavmeshTriangulation::getEdgeConstraintData(const MarkerType edgeMarker) const {
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

  auto getSuccessorStateThroughEdgeIfPossible = [this, &currentState, agentRadius](const IndexType entryEdgeIndex, const IndexType neighborTriangleIndex) -> std::optional<State> {
    if (currentState.hasEntryEdgeIndex() && entryEdgeIndex == currentState.getEntryEdgeIndex()) {
      // TODO: In the takla bridge case, it would make sense to return through the edge which we entered, only if the resulting state is different than the previous state
      // TODO: To solve this, maybe we need to have access to our previous state too?
      //  That info might be hard to get
      // TODO: Maybe we just dont do this check, and let the using algorithm filter out already-seen states
      // Don't return through the edge which we entered through
      return {};
    }

    if (!agentFitsThroughEdge(entryEdgeIndex, agentRadius)) {
      // Agent cannot fit through this edge, no successor this way
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
      const auto edgeConstraintData = getEdgeConstraintData(edgeMarker);
      if (edgeConstraintData.forTerrain() && edgeConstraintData.is(EdgeConstraintFlag::kGlobal)) {
        // Global edge of the region
        // Whether we're on an object or on the terrain, this edge has the same implications
        // TODO: Not handled
        //  TODO: Check if region is enabled
        //  TODO: Somehow return a successor that is on a different region
        // For now, we dont step off the region, return no successor
        return {};
      } else {
        if (currentState.isOnObject()) {
          // We are currently on some object
          if (edgeConstraintData.forObject()) {
            // This edge is a constraint of an object (and not a constraint of the terrain, again, might not be blocking)
            if (currentState.getObjectData() == edgeConstraintData.getObjectData()) {
              // This edge is a constraint of the object that we are on and the same area in that object that we're on
              if (edgeConstraintData.is(EdgeConstraintFlag::kBlocking)) {
                // Blocking edge for our object, cannot cross; no successor
                return {};
              } else if (edgeConstraintData.is(EdgeConstraintFlag::kGlobal)) {
                // External edge of the object
                if (edgeConstraintData.is(EdgeConstraintFlag::kBridge)) {
                  // From within the object, external bridge edges are blocking; no successor
                  return {};
                } else {
                  // This edge is non-blocking and is an outline edge of the object
                  //  Crossing this edge will lead us off the object
                  if (blockedTerrainTriangles_[neighborTriangleIndex]) {
                    // Trying to leave the object onto blocked terrain; no successor
                    return {};
                  } else {
                    // Leaving the object onto valid terrain
                    // Create a state which is on the terrain, in the new triangle, and with the entry edge
                    return State{neighborTriangleIndex, entryEdgeIndex};
                  }
                }
              } else {
                // Edge is not blocked and is an internal edge of our current object and current area (and is a constraint edge)
                //  These shouldn't even exit (because it shouldnt be a constraint edge)
                throw std::runtime_error("Edge is not blocked and is an internal edge of our current object (and is a constraint edge)");
              }
            } else {
              // This edge is a constraint of some other object or a different area of our current object
              //  Collision is not checked against objects other than the one we're on or other areas
              State newState = currentState;
              // Create a state which is still on this same object, in the new triangle, and with the entry edge
              newState.setNewTriangleAndEntryEdge(neighborTriangleIndex, entryEdgeIndex);
              return newState;
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
          if (edgeConstraintData.forObject()) {
            // This edge is a constraint of an object (and not a constraint of the terrain, again, might not be blocking)
            // This edge is a constraint of the object that we are on
            if (edgeConstraintData.is(EdgeConstraintFlag::kGlobal)) {
              // Only check collision against external edges of objects
              if (edgeConstraintData.is(EdgeConstraintFlag::kBlocking)) {
                // Cannot pass through; no successor
                return {};
              } else if (edgeConstraintData.is(EdgeConstraintFlag::kBridge)) {
                // Can "pass through", we stay on terrain because we're going under some kind of bridge
                // Create a state which is (still) on the terrain, in the new triangle, and with the entry edge
                return State{neighborTriangleIndex, entryEdgeIndex};
              } else {
                // Unblocked external edge of object
                bool weAreUnderTheObject{false};
                const auto &objectDatasForThisTriangle = getObjectDatasForTriangle(currentState.getTriangleIndex());
                if (std::find_if(objectDatasForThisTriangle.begin(), objectDatasForThisTriangle.end(), [&edgeConstraintData](const auto &objectData){
                  return (objectData.objectInstanceId == edgeConstraintData.getObjectData().objectInstanceId);
                }) != objectDatasForThisTriangle.end()) {
                  // We are on the terrain, but this triangle also overlaps with the same object as is referenced by this edge
                  //  We must be underneath the object
                  const auto &objectDatasForNeighborTriangle = getObjectDatasForTriangle(neighborTriangleIndex);
                  if (std::find_if(objectDatasForNeighborTriangle.begin(), objectDatasForNeighborTriangle.end(), [&edgeConstraintData](const auto &objectData){
                    return (objectData.objectInstanceId == edgeConstraintData.getObjectData().objectInstanceId);
                  }) == objectDatasForNeighborTriangle.end()) {
                    // The neighbor triangle does not overlap with this object, we must be coming out from underneath the object
                    weAreUnderTheObject = true;
                  }
                }
                if (weAreUnderTheObject) {
                  // Create a state which is still on the terrain, in the new triangle, and with the entry edge
                  State newState{neighborTriangleIndex, entryEdgeIndex};
                  return newState;
                } else {
                  // Create a state which is now on this object, in the new triangle, and with the entry edge
                  State newState{neighborTriangleIndex, entryEdgeIndex};
                  newState.setObjectData(edgeConstraintData.getObjectData());
                  return newState;
                }
              }
            } else {
              // On the terrain, internal edge of an object, I think collision is not checked in this case, even if blocking
              //  Can pass through
              // Create a state which is (still) on the terrain, in the new triangle, and with the entry edge
              return State{neighborTriangleIndex, entryEdgeIndex};
            }
          } else {
            // Must be constrained by the terrain
            //  Global edges are already handled, so it must be an internal edge
            if (!edgeConstraintData.is(EdgeConstraintFlag::kBlocking)) {
              // Can pass through
              // Create a state which is (still) on the terrain, in the new triangle, and with the entry edge
              return State{neighborTriangleIndex, entryEdgeIndex};
            } else {
              // Cannot pass through; no successor
              return {};
            }
          }
        }
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
      const auto edgeConstraintData = getEdgeConstraintData(edgeMarker);

      if (currentState.isOnObject()) {
        // We are currently on some object
        if (edgeConstraintData.forObject()) {
          // This edge is a constraint of an object (and not a constraint of the terrain, again, might not be blocking)
          if (currentState.getObjectData() == edgeConstraintData.getObjectData()) {
            // This edge is a constraint of the object that we are on and the same area in that object that we're on
            if (edgeConstraintData.is(EdgeConstraintFlag::kGlobal)) {
              // External edge of object; cannot leave object
              return {};
            } else if (edgeConstraintData.is(EdgeConstraintFlag::kInternal)) {
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
            return newState;
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

pathfinder::Vector SingleRegionNavmeshTriangulation::to2dPoint(const math::Vector &point) {
  // Convert our 3d point into the pathfinder's 2d point type
  return {point.x, point.z};
}

SingleRegionNavmeshTriangulation::State SingleRegionNavmeshTriangulation::createStateForPoint(const math::Vector &point, const IndexType triangleIndex) const {
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

SingleRegionNavmeshTriangulation::State SingleRegionNavmeshTriangulation::createStartState(const math::Vector &point, const IndexType triangleIndex) const {
  return createStateForPoint(point, triangleIndex);
}

SingleRegionNavmeshTriangulation::State SingleRegionNavmeshTriangulation::createGoalState(const math::Vector &point, const IndexType triangleIndex) const {
  auto state = createStateForPoint(point, triangleIndex);
  state.setIsGoal(true);
  return state;
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

} // namespace triangulation

} // namespace navmesh
