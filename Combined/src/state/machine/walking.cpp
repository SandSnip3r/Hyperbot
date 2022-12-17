#include "walking.hpp"

#include "bot.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"

#include "pathfinder.h"

#include <silkroad_lib/game_constants.h>
#include <silkroad_lib/position_math.h>

namespace state::machine {

Walking::Walking(Bot &bot, const sro::Position &destinationPosition) : StateMachine(bot) {
  stateMachineCreated(kName);
  waypoints_ = calculatePathToDestination(destinationPosition);
  pushBlockedOpcode(packet::Opcode::kClientAgentCharacterMoveRequest);
  LOG() << "Walking, with " << waypoints_.size() << " waypoint(s)" << std::endl;
}

Walking::~Walking() {
  stateMachineDestroyed();
}

void Walking::onUpdate(const event::Event *event) {
  LOG() << "Walking" << std::endl;
  if (done()) {
    return;
  }

  if (event) {
    if (const auto *movementBeganEvent = dynamic_cast<const event::EntityMovementBegan*>(event); movementBeganEvent != nullptr && movementBeganEvent->globalId == bot_.selfState().globalId) {
      // We started to move, our movement request must've been successful
      if (movementRequestTimeoutEventId_) {
        bot_.eventBroker().cancelDelayedEvent(*movementRequestTimeoutEventId_);
        movementRequestTimeoutEventId_.reset();
      } else {
        LOG() << "Movement began, but had no running movement request timeout timer" << std::endl;
      }
      // Nothing else to do here. We're now waiting for our movement to end
      return;
    } else if (const auto *movementEndedEvent = dynamic_cast<const event::EntityMovementEnded*>(event); movementEndedEvent != nullptr && movementEndedEvent->globalId == bot_.selfState().globalId) {
      // If we send a request to move, but get knocked back before the MovementBegin happens, the knockback movement will send this MovementEnded event
      if (movementRequestTimeoutEventId_) {
        bot_.eventBroker().cancelDelayedEvent(*movementRequestTimeoutEventId_);
        movementRequestTimeoutEventId_.reset();
      }
    } else if (event->eventCode == event::EventCode::kMovementRequestTimedOut) {
      LOG() << "Movement request timed out" << std::endl;
      movementRequestTimeoutEventId_.reset();
    }
  }

  if (bot_.selfState().moving()) {
    // Still moving, nothing to do
    LOG() << "Moving" << std::endl;
    return;
  }

  // We're not moving
  // Did we just arrive at this waypoint?
  while (currentWaypointIndex_ < waypoints_.size() && sro::position_math::calculateDistance2d(bot_.selfState().position(), waypoints_.at(currentWaypointIndex_)) < 1.0) { // TODO: Choose better measure of "close enough"
    // Already at this waypoint, increment index
    ++currentWaypointIndex_;
  }
  if (done()) {
    // Finished walking
    return;
  }

  // Not yet done walking
  if (movementRequestTimeoutEventId_) {
    // Already asked to move, nothing to do
    LOG() << "Waiting on requested movement" << std::endl;
    return;
  }

  if (!canMove()) {
    LOG() << "Can't move right now; nothing to do" << std::endl;
    return;
  }

  // We are not moving, we're not at the current waypoint, and there's not a pending movement request
  // Send a request to move to the current waypoint
  LOG() << "Requesting movement" << std::endl;
  const auto &currentWaypoint = waypoints_.at(currentWaypointIndex_);
  const auto movementPacket = packet::building::ClientAgentCharacterMoveRequest::moveToPosition(currentWaypoint);
  bot_.packetBroker().injectPacket(movementPacket, PacketContainer::Direction::kClientToServer);
  const int kMovementRequestTimeoutMs{333}; // TODO: Move somewhere else and make an educated guess about what this value should be
  movementRequestTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementRequestTimedOut), std::chrono::milliseconds(kMovementRequestTimeoutMs));
}

bool Walking::done() const {
  return (currentWaypointIndex_ == waypoints_.size());
}

std::vector<sro::Position> Walking::calculatePathToDestination(const sro::Position &destinationPosition) const {
  if (sro::position_math::calculateDistance2d(bot_.selfState().position(), destinationPosition) < 1.0) {
    // Already at destination
    return {};
  }
  auto pathfindingResultPathToVectorOfPositions = [&, this](const auto &pathfindingShortestPath) {
    const auto &navmeshTriangulation = bot_.gameData().navmeshTriangulation();

    // Get a list of all straight segments
    std::vector<pathfinder::StraightPathSegment*> straightSegments;
    for (const auto &segment : pathfindingShortestPath) {
      pathfinder::StraightPathSegment *straightSegment = dynamic_cast<pathfinder::StraightPathSegment*>(segment.get());
      if (straightSegment != nullptr) {
        straightSegments.push_back(straightSegment);
      }
    }

    // Turn straight segments into a list of waypoints
    std::vector<sro::Position> waypoints;
    // Note: We are ignoring the start of the first segment, since we assume we're already there
    for (int i=0; i<straightSegments.size()-1; ++i) {
      // Find the average between the end of this straight segment and the beginning of the next
      //  Between these two is an arc, which we're ignoring
      // TODO: There is a chance that this yields a bad path
      const auto &point1 = straightSegments[i]->endPoint;
      const auto &point2 = straightSegments[i+1]->startPoint;
      const auto midpoint = pathfinder::math::extendLineSegmentToLength(point1, point2, pathfinder::math::distance(point1, point2)/2.0);
      const auto regionAndPointPair = navmeshTriangulation.transformAbsolutePointIntoRegion({static_cast<float>(midpoint.x()), 0.0f, static_cast<float>(midpoint.y())});
      // TODO: Rounding the position could result in an invalid path
      waypoints.emplace_back(regionAndPointPair.first, std::round(regionAndPointPair.second.x), std::round(regionAndPointPair.second.y), std::round(regionAndPointPair.second.z));
    }

    // Additionally, add the endpoint of the final segment
    const auto finalRegionAndPointPair = navmeshTriangulation.transformAbsolutePointIntoRegion({static_cast<float>(straightSegments.back()->endPoint.x()), 0.0f, static_cast<float>(straightSegments.back()->endPoint.y())});
    waypoints.emplace_back(finalRegionAndPointPair.first, std::round(finalRegionAndPointPair.second.x), std::round(finalRegionAndPointPair.second.y), std::round(finalRegionAndPointPair.second.z));

    // Remove duplicates
    auto newEndIt = std::unique(waypoints.begin(), waypoints.end());
    if (newEndIt != waypoints.end()) {
      LOG() << "Removed " << std::distance(newEndIt, waypoints.end()) << " duplicate waypoints" << std::endl;
      waypoints.erase(newEndIt, waypoints.end());
    }
    return waypoints;
  };

  auto breakUpLongMovements = [](std::vector<sro::Position> &waypoints) {
    auto tooFar = [](const auto &srcWaypoint, const auto &destWaypoint) {
      // The difference between a pair of xOffsets must be <= 1920.
      // The difference between a pair of zOffsets must be <= 1920.
      const auto [dx, dz] = sro::position_math::calculateOffset2d(srcWaypoint, destWaypoint);
      return (std::abs(dx) > sro::game_constants::kRegionWidth ||
              std::abs(dz) > sro::game_constants::kRegionHeight);
    };
    auto splitWaypoints = [](const auto &srcWaypoint, const auto &destWaypoint) {
      const auto [dx, dz] = sro::position_math::calculateOffset2d(srcWaypoint, destWaypoint);
      if (std::abs(dx) > std::abs(dz)) {
        const double ratio = static_cast<double>(sro::game_constants::kRegionWidth) / std::abs(dx);
        const auto newDxOffset = (dx > 0 ? 1 : -1) * sro::game_constants::kRegionWidth;
        const double newDzOffset = dz * ratio;
        return sro::position_math::createNewPositionWith2dOffset(destWaypoint, -newDxOffset, -newDzOffset);
      } else {
        const double ratio = static_cast<double>(sro::game_constants::kRegionWidth) / std::abs(dz);
        const auto newDxOffset = dx * ratio;
        const double newDzOffset = (dz > 0 ? 1 : -1) * sro::game_constants::kRegionWidth;
        return sro::position_math::createNewPositionWith2dOffset(destWaypoint, -newDxOffset, -newDzOffset);
      }
    };
    for (int i=waypoints.size()-1; i>0;) {
      if (tooFar(waypoints.at(i-1), waypoints.at(i))) {
        // Pick a point that is the maxmimum distance possible away from waypoints[i] and insert it before waypoints[i]
        const auto newWaypoint = splitWaypoints(waypoints.at(i-1), waypoints.at(i));
        waypoints.insert(waypoints.begin()+i, newWaypoint);
      } else {
        --i;
      }
    }
  };

  constexpr const double kAgentRadius{7.23};
  pathfinder::Pathfinder<navmesh::triangulation::NavmeshTriangulation> pathfinder(bot_.gameData().navmeshTriangulation(), kAgentRadius);
  try {
    const auto currentPosition = bot_.selfState().position();
    const math::Vector currentPositionPoint(currentPosition.xOffset(), currentPosition.yOffset(), currentPosition.zOffset());
    const auto navmeshCurrentPosition = bot_.gameData().navmeshTriangulation().transformRegionPointIntoAbsolute(currentPositionPoint, currentPosition.regionId());

    const math::Vector destinationPositionPoint(destinationPosition.xOffset(), destinationPosition.yOffset(), destinationPosition.zOffset());
    const auto navmeshDestinationPosition = bot_.gameData().navmeshTriangulation().transformRegionPointIntoAbsolute(destinationPositionPoint, destinationPosition.regionId());

    const auto pathfindingResult = pathfinder.findShortestPath(navmeshCurrentPosition, navmeshDestinationPosition);
    const auto &path = pathfindingResult.shortestPath;
    if (path.empty()) {
      throw std::runtime_error("Found empty path");
    }
    auto waypoints = pathfindingResultPathToVectorOfPositions(path);
    // Add our own position to the beginning of this list so that we can break up the distance if it's too far.
    waypoints.insert(waypoints.begin(), currentPosition);
    breakUpLongMovements(waypoints);
    return waypoints;
  } catch (std::exception &ex) {
    throw std::runtime_error("Cannot find path with pathfinder: \""+std::string(ex.what())+"\"");
  }
}

} // namespace state::machine