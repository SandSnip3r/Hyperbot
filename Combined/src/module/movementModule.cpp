#include "movementModule.hpp"
#include "../math/position.hpp"
#include "../packet/opcode.hpp"
#include "../packet/building/clientAgentCharacterMoveRequest.hpp"
#include "../packet/building/serverAgentChatUpdate.hpp"

#include "pathfinder.h"

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <regex>

#define LOG_TO_STREAM(OSTREAM, TAG) (OSTREAM) << '[' << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() << "] " << (#TAG) << ": "
#define LOG(TAG) LOG_TO_STREAM(std::cout, TAG)

// TODO: Remove
#define SHARED_MEM 1

std::ostream& operator<<(std::ostream &stream, const packet::structures::Position &pos) {
  stream << '{';
  if (pos.isDungeon()) {
    stream << (int)pos.dungeonId();
  } else {
    stream << (int)pos.xSector() << ',' << (int)pos.zSector();
  }
  stream << " (" << pos.xOffset << ',' << pos.yOffset << ',' << pos.zOffset << ")}";
  return stream;
}

std::mt19937 createRandomEngine() {
  std::random_device rd;
  std::array<int, std::mt19937::state_size> seed_data;
  std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  return std::mt19937(seq);
}

namespace module {

MovementModule::MovementModule(state::Entity &entityState,
                               state::Self &selfState,
                               storage::Storage &inventory,
                               broker::PacketBroker &brokerSystem,
                               broker::EventBroker &eventBroker,
                               const packet::parsing::PacketParser &packetParser,
                               const pk2::GameData &gameData) :
      entityState_(entityState),
      selfState_(selfState),
      inventory_(inventory),
      broker_(brokerSystem),
      eventBroker_(eventBroker),
      packetParser_(packetParser),
      gameData_(gameData) {
  auto packetHandleFunction = std::bind(&MovementModule::handlePacket, this, std::placeholders::_1);
  // Client packets
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentChatRequest, packetHandleFunction);
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentCharacterMoveRequest, packetHandleFunction);
  // Server packets
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionSelectResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMovement, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdatePosition, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntitySyncPosition, packetHandleFunction);
  // TODO: When teleporting, cancel movement timer!!

  auto eventHandleFunction = std::bind(&MovementModule::handleEvent, this, std::placeholders::_1);
  eventBroker_.subscribeToEvent(event::EventCode::kMovementEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterSpeedUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRepublish, eventHandleFunction);
#ifdef SHARED_MEM
  eventBroker_.subscribeToEvent(event::EventCode::kTemp, eventHandleFunction);
#endif

  eng_ = createRandomEngine();

#ifdef SHARED_MEM
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kTemp));
#endif
}

bool MovementModule::handlePacket(const PacketContainer &packet) {
  std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    std::cerr << "[MovementModule] Failed to parse packet " << std::hex << packet.opcode << std::dec << "\n  Error: \"" << ex.what() << "\"\n";
    return true;
  }

  if (!parsedPacket) {
    // Not yet parsing this packet
    return true;
  }

  auto *clientChat = dynamic_cast<packet::parsing::ClientAgentChatRequest*>(parsedPacket.get());
  if (clientChat != nullptr) {
    return clientAgentChatRequestReceived(*clientChat);
  }

  auto *clientMove = dynamic_cast<packet::parsing::ClientAgentCharacterMoveRequest*>(parsedPacket.get());
  if (clientMove != nullptr) {
    return clientAgentCharacterMoveRequestReceived(*clientMove);
  }

  auto *severMove = dynamic_cast<packet::parsing::ServerAgentEntityUpdateMovement*>(parsedPacket.get());
  if (severMove != nullptr) {
    return serverAgentEntityUpdateMovementReceived(*severMove);
  }

  auto *entityUpdatePosition = dynamic_cast<packet::parsing::ServerAgentEntityUpdatePosition*>(parsedPacket.get());
  if (entityUpdatePosition != nullptr) {
    return serverAgentEntityUpdatePositionReceived(*entityUpdatePosition);
  }

  auto *entitySyncPosition = dynamic_cast<packet::parsing::ServerAgentEntitySyncPosition*>(parsedPacket.get());
  if (entitySyncPosition != nullptr) {
    return serverAgentEntitySyncPositionReceived(*entitySyncPosition);
  }


  std::cout << "MovementModule: Unhandled packet subscribed to\n";
  return true;
}

void MovementModule::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
  const auto eventCode = event->eventCode;
  switch (eventCode) {
    case event::EventCode::kMovementEnded:
      handleMovementEnded();
      break;
    case event::EventCode::kCharacterSpeedUpdated:
      handleSpeedUpdated();
      break;
    case event::EventCode::kTemp:
      handleTempEvent();
      break;
    case event::EventCode::kRepublish:
      republishStepEventId_.reset();
      if (!selfState_.moving()) {
        if (republishCount_ < 50) { //15==1s max
          LOG(handleEvent) << "<<<<<<<<<<<<<<<<<<<<<Triggering \"republish\">>>>>>>>>>>>>>>>>>>>>\n";
          ++republishCount_;
          takeNextStepOnPath();
        } else {
          LOG(handleEvent) << "Republishing timed out. Stopping\n";
          stopAutowalkTest();
        }
      }
      break;
    default:
      std::cout << "Unhandled event subscribed to. Code:" << static_cast<int>(eventCode) << '\n';
      break;
  }
}

void MovementModule::handleTempEvent() {
#ifdef SHARED_MEM
  auto pos = selfState_.position();
  // Write pos at beginning of file
  sharedMemoryWriter_.seek(0);
  sharedMemoryWriter_.writeData(pos.regionId);
  sharedMemoryWriter_.writeData(pos.xOffset);
  sharedMemoryWriter_.writeData(pos.yOffset);
  sharedMemoryWriter_.writeData(pos.zOffset);
  eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kTemp), std::chrono::milliseconds(33)); // 30FPS
#endif
}

void MovementModule::handleSpeedUpdated() {
  if (movingEventId_ && selfState_.moving()) {
    if (selfState_.haveDestination()) {
      // Need to update timer
      auto seconds = secondsToTravel(selfState_.position(), selfState_.destination());
      eventBroker_.cancelDelayedEvent(*movingEventId_);
      movingEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
    }
  }
}

void MovementModule::handleMovementEnded() {
  movingEventId_.reset();
  LOG(handleMovementEnded) << "Movement ended event\n";
  selfState_.doneMoving();
  LOG(handleMovementEnded) << "Currently at " << selfState_.position() << '\n';

  if (!waypoints_.empty()) {
    const auto &nextWaypoint = waypoints_.front();
    const auto &currentPosition = selfState_.position();
    if (std::round(currentPosition.xOffset) == std::round(nextWaypoint.xOffset) && std::round(currentPosition.zOffset) == std::round(nextWaypoint.zOffset)) {
      LOG(handleMovementEnded) << "Walking on a path, arrived at the next waypoint" << std::endl;
      waypoints_.erase(waypoints_.begin());
    } else {
      // Didn't arrive at next waypoint. This can happen if we prematurely resend the movement packet
    }
  }

  if (testingAutowalk_) {
    distanceTraveled_ += queuedMovementDistance_;
    LOG(handleMovementEnded) << "Total distance traveled: " << distanceTraveled_ << '\n';
  }
  if (!waypoints_.empty()) {
    LOG(handleMovementEnded) << "Arrived at waypoint, moving to next one\n";
    LOG(handleMovementEnded) << "  Remaining points: [";
    for (const auto &i : waypoints_) {
      std::cout << '(' << i.regionId << ',' << i.xOffset << ',' << i.yOffset << ',' << i.zOffset << ") ";
    }
    std::cout << ']' << std::endl;
    takeNextStepOnPath();
  } else {
    // No waypoints left
    if (testingAutowalk_) {
      // Path somewhere else now
      pathToRandomPoint();
    }
  }
}

bool MovementModule::serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet) {
  if (packet.globalId() == selfState_.globalId()) {
    const auto &currentPosition = selfState_.position();
    LOG(serverAgentEntitySyncPositionReceived) << "Syncing position: " << currentPosition.xOffset << ',' << currentPosition.zOffset << std::endl;
    selfState_.syncPosition(packet.position());
  }
  return false;
}

bool MovementModule::serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) {
  if (packet.globalId() == selfState_.globalId()) {
    if (selfState_.moving()) {
      LOG(serverAgentEntityUpdatePositionReceived) << "Position update received" << std::endl;
      // Happens when you collide with something
      // Note: this also happens when running to pick an item
      // Note: I think this also happens when a speed drug is cancelled
      if (republishStepEventId_) {
        LOG(serverAgentEntityUpdatePositionReceived) << "Cancelling republish timer\n";
        eventBroker_.cancelDelayedEvent(*republishStepEventId_);
        republishStepEventId_.reset();
      }
      if (movingEventId_) {
        LOG(serverAgentEntityUpdatePositionReceived) << "Cancelling movement timer\n";
        eventBroker_.cancelDelayedEvent(*movingEventId_);
        movingEventId_.reset();
      }
      selfState_.setPosition(packet.position());
      LOG(serverAgentEntityUpdatePositionReceived) << "Now stationary at " << selfState_.position() << '\n';
      if (testingAutowalk_) {
        if (!waypoints_.empty()) {
          const auto &nextWaypoint = waypoints_.front();
          if (std::round(packet.position().xOffset) == std::round(nextWaypoint.xOffset) && std::round(packet.position().zOffset) == std::round(nextWaypoint.zOffset)) {
            // We've essentially arrived at our waypoint. Good
            LOG(serverAgentEntityUpdatePositionReceived) << "Arrived at waypoint, moving to next one\n";
            waypoints_.erase(waypoints_.begin());
            takeNextStepOnPath();
          } else {
            LOG(serverAgentEntityUpdatePositionReceived) << "We were autowalking, this is a problem!!!\n";
            stopAutowalkTest();
          }
        } else {
          LOG(serverAgentEntityUpdatePositionReceived) << "We were autowalking but it seems that we have no current waypoint...\n";
        }
      } else if (!waypoints_.empty()) {
        LOG(serverAgentEntityUpdatePositionReceived) << "Have a non-empty waypoint list. Clearing\n";
        waypoints_.clear();
      }
    } else {
      LOG(serverAgentEntityUpdatePositionReceived) << "We werent moving, weird\n";
      const auto pos = selfState_.position();
      LOG(serverAgentEntityUpdatePositionReceived) << "Expected pos: " << pos.xOffset << ',' << pos.zOffset << '\n';
      LOG(serverAgentEntityUpdatePositionReceived) << "Received pos: " << packet.position().xOffset << ',' << packet.position().zOffset << '\n';
    }
  }
  return true;
}

bool MovementModule::clientAgentCharacterMoveRequestReceived(packet::parsing::ClientAgentCharacterMoveRequest &packet) {
  if (testingAutowalk_) {
    // Discard request, dont want client to interrupt
    // Tell the user through chat
    broker_.injectPacket(packet::building::ServerAgentChatUpdate::notice("Autowalk Testing in progress. Movement input ignored."), PacketContainer::Direction::kServerToClient);
    return false;
  }
  // std::cout << "Character is moving\n";
  // if (packet.hasDestination()) {
  //   std::cout << "  Has destination\n";
  //   if (packet.regionId() & 0x8000) {
  //     // Dungeon
  //     uint16_t regionId = packet.regionId() & 0x7FFF;
  //     std::cout << "    Dungeon: region(" << regionId << ") " << packet.dungeonXOffset() << ',' << packet.dungeonYOffset() << ',' << packet.dungeonZOffset() << '\n';
  //   } else {
  //     uint8_t xSector = packet.regionId() & 0xFF;
  //     uint8_t zSector = (packet.regionId()>>8) & 0x7F;
  //     std::cout << "    World: region(" << (int)xSector << ',' << (int)zSector << ")" << packet.worldXOffset() << ',' << packet.worldYOffset() << ',' << packet.worldZOffset() << '\n';
  //     const auto posX = (xSector - 135)*192 + packet.worldXOffset()/10.0;
  //     const auto posZ = (zSector - 92)*192 + packet.worldZOffset()/10.0;
  //     std::cout << "    World coords: " << posX << ',' << posZ << '\n';
  //   }
  // } else {
  //   std::cout << "  No destination\n";
  //   std::cout << "    Angle: " << packet.angle() << '\n';
  // }
  return true;
}

float MovementModule::secondsToTravel(const packet::structures::Position &srcPosition, const packet::structures::Position &destPosition) const {
  auto distance = math::position::calculateDistance(srcPosition, destPosition);
  return distance / selfState_.currentSpeed();
}

bool MovementModule::serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) {
  if (packet.globalId() == selfState_.globalId()) {
    // If there's a timer running to republish, cancel because our message was received
    if (republishStepEventId_) {
      eventBroker_.cancelDelayedEvent(*republishStepEventId_);
      republishStepEventId_.reset();
    }
    // If we republished, it worked. Reset count
    republishCount_ = 0;
    {
      std::ofstream file("republish_delay.txt", std::ios::app);
      LOG_TO_STREAM(file, serverAgentEntityUpdateMovementReceived) << "Republishing worked\n" << std::endl;
    }

    packet::structures::Position sourcePosition;
    if (packet.hasSource()) {
      // Server is telling us our source position
      sourcePosition = packet.sourcePosition();
      const auto currentPosition = selfState_.position();
      if (std::round(currentPosition.xOffset) != std::round(sourcePosition.xOffset) && std::round(currentPosition.zOffset) != std::round(sourcePosition.zOffset)) {
        // We arent where we thought we were
        // We need to cancel this movement. Either move back to the source position or try to stop exactly where we are (by moving to our estimated position)
        LOG(serverAgentEntityUpdateMovementReceived) << "Whoa, we're a bit off from where we thought we were. Expected: " << currentPosition.xOffset << ',' << currentPosition.zOffset << ", actual: " << sourcePosition.xOffset << ',' << sourcePosition.zOffset << std::endl;
      }
      LOG(serverAgentEntityUpdateMovementReceived) << "Syncing src position: " << sourcePosition.xOffset << ',' << sourcePosition.zOffset << std::endl;
      selfState_.syncPosition(sourcePosition);
    } else {
      // Server doesnt tell us where we're coming from, use our internally tracked position
      sourcePosition = selfState_.position();
    }
    if (movingEventId_) {
      // Had a timer already running for movement, cancel it
      LOG(serverAgentEntityUpdateMovementReceived) << "Had a running timer, cancelling it" << std::endl;
      eventBroker_.cancelDelayedEvent(*movingEventId_);
      movingEventId_.reset();
    }
    LOG(serverAgentEntityUpdateMovementReceived) << "We are moving from " << sourcePosition << ' ';
    if (packet.hasDestination()) {
      auto destPosition = packet.destinationPosition();
      std::cout << "to " << destPosition << '\n';
      if (sourcePosition.xOffset == destPosition.xOffset && sourcePosition.zOffset == destPosition.zOffset) {
        LOG(serverAgentEntityUpdateMovementReceived) << "Server says we're moving to our current position. wtf?\n";
        // Ignore this
        if (!waypoints_.empty()) {
          const auto &nextWaypoint = waypoints_.front();
          if (std::round(destPosition.xOffset) == std::round(nextWaypoint.xOffset) && std::round(destPosition.zOffset) == std::round(nextWaypoint.zOffset)) {
            // This is the next waypoint though, pop it off now and take the next step
            LOG(serverAgentEntityUpdateMovementReceived) << "This is actually our next waypoint though, pop and move to next one\n";
            waypoints_.erase(waypoints_.begin());
          } else {
            LOG(serverAgentEntityUpdateMovementReceived) << "This is not our next waypoint. I'd guess that this was our previous waypoint that we sent twice. Anyways, continue\n";
          }
          takeNextStepOnPath();
        } else {
          LOG(serverAgentEntityUpdateMovementReceived) << " No waypoints either.\n";
        }
      } else {
        queuedMovementDistance_ = math::position::calculateDistance(sourcePosition, destPosition);
        auto seconds = secondsToTravel(sourcePosition, destPosition);
        LOG(serverAgentEntityUpdateMovementReceived) << "Should take " << seconds << "s. Timer set\n";
        movingEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
        selfState_.setMoving(packet.destinationPosition());
      }
    } else {
      std::cout << "toward " << packet.angle() << '\n';
      selfState_.setMoving(packet.angle());
    }
  }
  return true;
}

void MovementModule::takeNextStepOnPath() {
  if (waypoints_.empty()) {
    LOG(takeNextStepOnPath) << "Trying to take a step but there's no path!\n";
    if (testingAutowalk_) {
      pathToRandomPoint();
    }
    return;
  }
  const auto &currentPos = selfState_.position();
  auto isSameAsCurrentPosition = [&currentPos](const auto &waypoint) {
    if (std::round(currentPos.xOffset) == std::round(waypoint.xOffset) && std::round(currentPos.zOffset) == std::round(waypoint.zOffset)) {
      return true;
    } else {
      return false;
    }
  };
  LOG(takeNextStepOnPath) << "takeNextStepOnPath: Current position " << currentPos.xOffset << ',' << currentPos.zOffset << '\n';
  while (!waypoints_.empty() && isSameAsCurrentPosition(waypoints_.front())) {
    LOG(takeNextStepOnPath) << "  pop off " << waypoints_.front().regionId << ',' << waypoints_.front().xOffset << ',' << waypoints_.front().zOffset << '\n';
    waypoints_.erase(waypoints_.begin());
  }
  if (waypoints_.empty()) {
    LOG(takeNextStepOnPath) << "Looks like we're already at our goal\n";
    if (testingAutowalk_) {
      pathToRandomPoint();
    }
    return;
  }
  const auto nextWaypoint = waypoints_.front();
  LOG(takeNextStepOnPath) << "Sending movement packet to position " << nextWaypoint.regionId << ',' << std::round(nextWaypoint.xOffset) << ',' << std::round(nextWaypoint.zOffset) << ", " << waypoints_.size() << " steps left\n";
  broker_.injectPacket(packet::building::ClientAgentCharacterMoveRequest::packet(nextWaypoint.regionId, std::round(nextWaypoint.xOffset), nextWaypoint.yOffset, std::round(nextWaypoint.zOffset)), PacketContainer::Direction::kClientToServer);
  
  if (testingAutowalk_ && republishCount_ == 0) {
    // replayPoints_.emplace_back(nextWaypoint.x, nextWaypoint.z); // TODO: Replace
    if (replayPoints_.size() > kReplayPointCount_) {
      replayPoints_.pop_front();
    }
  }

  if (republishStepEventId_) {
    // The last step was likely a quick one, cancel the old event
    eventBroker_.cancelDelayedEvent(*republishStepEventId_);
    republishStepEventId_.reset();
  }
  // TODO: This republish timeout should be proprotional to ping
  const double republishTimeout = 100.0 * pow(1.16591440117983, republishCount_);
  if (republishCount_ > 0) {
    std::cout << "Seting republish delay with higher value: " << republishTimeout << std::endl;
    std::ofstream file("republish_delay.txt", std::ios::app);
    LOG_TO_STREAM(file, takeNextStepOnPath) << "Increasing delay to " << republishTimeout << std::endl;
  }
  republishStepEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kRepublish), std::chrono::milliseconds(static_cast<uint64_t>(republishTimeout)));
}

void MovementModule::executePath(const std::vector<std::unique_ptr<pathfinder::PathSegment>> &segments) {
  if (!waypoints_.empty()) {
    LOG(executePath) << "WEIRD! Executing a path but there are already waypoints in the queue\n";
  }
  waypoints_.clear();
  bool addFirstPoint{false};
  const auto &navmeshTriangulation = gameData_.navmeshTriangulation();
  for (const auto &segment : segments) {
    pathfinder::StraightPathSegment *straightSegment = dynamic_cast<pathfinder::StraightPathSegment*>(segment.get());
    if (straightSegment != nullptr) {
      if (addFirstPoint) {
        const auto regionAndPointPair = navmeshTriangulation.transformAbsolutePointIntoRegion({static_cast<float>(straightSegment->startPoint.x()), 0.0f, static_cast<float>(straightSegment->startPoint.y())});
        waypoints_.push_back({regionAndPointPair.first, regionAndPointPair.second.x, regionAndPointPair.second.y, regionAndPointPair.second.z});
      }
      const auto regionAndPointPair = navmeshTriangulation.transformAbsolutePointIntoRegion({static_cast<float>(straightSegment->endPoint.x()), 0.0f, static_cast<float>(straightSegment->endPoint.y())});
      waypoints_.push_back({regionAndPointPair.first, regionAndPointPair.second.x, regionAndPointPair.second.y, regionAndPointPair.second.z});
      addFirstPoint = true;
    }
  }
  LOG(executePath) << "Path: [";
  for (const auto &i : waypoints_) {
    std::cout << '(' << i.regionId << ',' << i.xOffset << ',' << i.zOffset << ") ";
  }
  std::cout << "]\n";

  takeNextStepOnPath();
}

MovementModule::PathfindingResult MovementModule::pathToPosition(const pathfinder::Vector &position) {
  const auto currentPos = selfState_.position();
  try {
    const auto &navmeshTriangulation = gameData_.navmeshTriangulation();
    pathfinder::Pathfinder<navmesh::triangulation::NavmeshTriangulation> pathfinder(navmeshTriangulation, agentRadius_);
    const auto start = navmeshTriangulation.transformRegionPointIntoAbsolute({currentPos.xOffset, currentPos.yOffset, currentPos.zOffset}, currentPos.regionId);
    const auto goal = navmeshTriangulation.transformRegionPointIntoAbsolute({currentPos.xOffset, -1000, currentPos.zOffset}, currentPos.regionId);
    auto result = pathfinder.findShortestPath(start, goal);
    if (!result.shortestPath.empty()) {
      LOG(pathToPosition) << "Pathing from " << start.x << ',' << start.y << ',' << start.z << " to " << goal.x << ',' << goal.y << ',' << goal.z << "\n";
      executePath(result.shortestPath);
      return PathfindingResult::kSuccess;
    } else {
      LOG(pathToPosition) << "No path exists from " << start.x << ',' << start.y << ',' << start.z << " to " << goal.x << ',' << goal.y << ',' << goal.z << "\n";
      return PathfindingResult::kPathNotPosible;
    }
  } catch (std::exception &ex) {
    LOG(pathToPosition) << "Exception caught while finding path from " << currentPos.xOffset << ',' << currentPos.zOffset << " to " << position.x() << ',' << position.y() << "\n";
    LOG(pathToPosition) << "  \"" << ex.what() << "\"\n";
    const std::string exceptionText = ex.what();
    if (exceptionText.find("No point found!") != std::string::npos) {
      return PathfindingResult::kExceptionNoPointFound;
    } else if (exceptionText.find("The chosen start point is overlapping with a constraint") != std::string::npos) {
      return PathfindingResult::kExceptionStartOverlapsWithConstraint;
    } else {
      return PathfindingResult::kException;
    }
  }
}

void MovementModule::pathToRandomPoint() {
  std::uniform_int_distribution<> dist(1,1919);
  auto createRandomPoint = [this, &dist]() -> pathfinder::Vector {
    return {static_cast<double>(dist(eng_)), static_cast<double>(dist(eng_))};
  };
  bool foundValidPoint{false};
  const auto currentPos = selfState_.position();
  do {
    const auto goalPoint = createRandomPoint();
    auto result = pathToPosition(goalPoint);
    if (result == PathfindingResult::kSuccess) {
      foundValidPoint = true;
    } else if (result == PathfindingResult::kExceptionNoPointFound) {
      LOG(pathToRandomPoint) << "Cancelling autowalk test\n";
      stopAutowalkTest();
      return;
    } else if (result == PathfindingResult::kExceptionStartOverlapsWithConstraint) {
      LOG(pathToRandomPoint) << "Cancelling autowalk test\n";
      stopAutowalkTest();
      return;
    }
  } while (!foundValidPoint);
}

void MovementModule::startAutowalkTest() {
  LOG(startAutowalkTest) << "Starting autowalk test\n";
  distanceTraveled_ = 0;
  testingAutowalk_ = true;
  pathToRandomPoint();
}

void MovementModule::stopAutowalkTest() {
  LOG(stopAutowalkTest) << "Stopping autowalk test\n";
  testingAutowalk_ = false;
  waypoints_.clear();
  republishStepEventId_.reset();
  republishCount_ = 0;
}

bool MovementModule::clientAgentChatRequestReceived(packet::parsing::ClientAgentChatRequest &packet) {
  std::regex printReplaySequenceRegex(R"delim(prs)delim"); // Print Replay Sequence
  std::regex replaceSequenceAtIndexRegex(R"delim(replay\s+([0-9]+))delim"); // Replay sequence starting at given index
  std::regex moveRegex(R"delim(move\s+(-?[0-9]+)\s+(-?[0-9]+))delim");
  std::regex move3Regex(R"delim(move\s+(-?[0-9]+)\s+(-?[0-9]+)\s+(-?[0-9]+))delim");
  std::regex pathRegex(R"delim(path\s+(-?[0-9]+)\s+(-?[0-9]+))delim");
  std::regex testRegex(R"delim(test)delim");
  std::regex stopRegex(R"delim(stop)delim");
  std::smatch regexMatch;

  if (std::regex_match(packet.message(), regexMatch, moveRegex)) {
    const int x = std::stoi(regexMatch[1].str());
    const int y = std::stoi(regexMatch[2].str());
    const auto pos = selfState_.position();
    broker_.injectPacket(packet::building::ClientAgentCharacterMoveRequest::packet(pos.regionId, x, 0, y), PacketContainer::Direction::kClientToServer);
  } else if (std::regex_match(packet.message(), regexMatch, move3Regex)) {
    const int x = std::stoi(regexMatch[1].str());
    const int y = std::stoi(regexMatch[2].str());
    const int z = std::stoi(regexMatch[3].str());
    const auto pos = selfState_.position();
    broker_.injectPacket(packet::building::ClientAgentCharacterMoveRequest::packet(pos.regionId, x, y, z), PacketContainer::Direction::kClientToServer);
  } else if (std::regex_match(packet.message(), regexMatch, pathRegex)) {
    const int x = std::stoi(regexMatch[1].str());
    const int y = std::stoi(regexMatch[2].str());
    const auto pos = selfState_.position();
    std::cout << "Checking the shortest path from " << pos.xOffset << ',' << pos.zOffset << " to " << x << ',' << y << " in region " << pos.regionId << '\n';
    auto result = pathToPosition(pathfinder::Vector{static_cast<double>(x),static_cast<double>(y)});
    if (result == PathfindingResult::kSuccess) {
      LOG(clientAgentChatRequestReceived) << "Found a path!\n";
    } else {
      LOG(clientAgentChatRequestReceived) << "No path possible\n";
    }
    return true;
    // const int x = std::stoi(regexMatch[1].str());
    // const int y = std::stoi(regexMatch[2].str());
    // // Move character
    // std::cout << "Moving character to " << x << ',' << y << '\n';
    // // | 15 | 14 | 13 | 12 | 11 | 10 | 09 | 08 | 07 | 06 | 05 | 04 | 03 | 02 | 01 | 00 |
    // // | DF |            ZSector               |                XSector                |
    // // DF = Dungeon flag
    // // 6B 69                                             ki..............
    // // 76 07                                             v...............
    // // B4 00                                             ................
    // // 98 06
    // // -5186,2664
    // // 0b0 1101001 01101011
    // // No dungeon, 0x69 z sector, 0x6b x sector
    // // X offset 0x767
    // // Y offset 0xB4
    // // Z offset 0x698
    // struct XY {
    //   int x,y;
    // };
    // struct RegionXYZ {
    //   int regionId;
    //   int x,y,z;
    // };
    // auto converterFunc = [](const XY &obj) {
    //   RegionXYZ result;
    //   // byte ySector = (byte)(Region >> 8);
    //   // byte xSector = (byte)(Region & 0xFF);

    //   // if (Region < short.MaxValue) // At world map? convert it!
    //   // {
    //   //     // 192px 1:10 scale, center (135,92)
    //   //     PosX = (xSector - 135) * 192 + X / 10;
    //   //     PosY = (ySector - 92) * 192 + Y / 10;
    //   // }
    //   // result.regionId
    // };
    // const auto pos = selfState_.position();
    // broker_.injectPacket(packet::building::ClientAgentCharacterMoveRequest::packet(pos.regionId, x, 0, y), PacketContainer::Direction::kClientToServer);
    // return false;
  } else if (std::regex_match(packet.message(), regexMatch, testRegex)) {
    if (!testingAutowalk_) {
      startAutowalkTest();
    }
  } else if (std::regex_match(packet.message(), regexMatch, stopRegex)) {
    if (testingAutowalk_) {
      stopAutowalkTest();
    }
  } else if (std::regex_match(packet.message(), regexMatch, printReplaySequenceRegex)) {
    // std::regex printReplaySequenceRegex(R"delim(prs)delim"); // Print Replay Sequence
    broker_.injectPacket(packet::building::ServerAgentChatUpdate::packet(packet::enums::ChatType::kPm, "Hyperbot", "Replay Sequence:"), PacketContainer::Direction::kServerToClient);
    int index=0;
    for (const auto &waypoint : replayPoints_) {
      std::stringstream msgSs;
      msgSs << "  " << index << ". ";
      msgSs << std::setw(4) << static_cast<int>(waypoint.x()) << ",";
      msgSs << std::setw(4) << static_cast<int>(waypoint.y());
      broker_.injectPacket(packet::building::ServerAgentChatUpdate::packet(packet::enums::ChatType::kPm, "Hyperbot", msgSs.str()), PacketContainer::Direction::kServerToClient);
      ++index;
    }
    return false;
  } else if (std::regex_match(packet.message(), regexMatch, replaceSequenceAtIndexRegex)) {
    // std::regex replaceSequenceAtIndexRegex(R"delim(replay ([0-9]+))delim"); // Replay sequence starting at given index
    const size_t index = std::stoi(regexMatch[1].str());
    if (index < 0 || index >= replayPoints_.size()) {
      std::cout << "Asking to replay from an invalid index of the sequence" << std::endl;
    } else {
      if (!waypoints_.empty()) {
        std::cout << "Trying to replay while a sequence of steps is being executed!" << std::endl;
      } else {
        // auto it = std::next(replayPoints_.begin(), index);
        // while (it != replayPoints_.end()) {
        //   waypoints_.emplace_back(it->x(), 0, it->y());
        //   it = std::next(it);
        // } // TODO: Fix
        std::cout << "Waypoint list for replay built. Executing" << std::endl;
        takeNextStepOnPath();
      }
    }
    return false;
  }
  return true;
}

// void MovementModule::serverAgentActionSelectResponseReceived(packet::parsing::ServerAgentActionSelectResponse &packet) {}

} // namespace module