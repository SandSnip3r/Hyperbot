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
    default:
      std::cout << "Unhandled event subscribed to. Code:" << static_cast<int>(eventCode) << '\n';
      break;
  }
}

void MovementModule::handleSpeedUpdated() {
  if (selfState_.haveMovingEventId() && selfState_.moving()) {
    if (selfState_.haveDestination()) {
      // Need to update timer
      auto seconds = secondsToTravel(selfState_.position(), selfState_.destination());
      eventBroker_.cancelDelayedEvent(selfState_.getMovingEventId());
      const auto movingEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
      selfState_.setMovingEventId(movingEventId);
    }
  }
}

void MovementModule::handleMovementEnded() {
  selfState_.resetMovingEventId();
  LOG(handleMovementEnded) << "Movement ended event\n";
  selfState_.doneMoving();
  LOG(handleMovementEnded) << "Currently at " << selfState_.position() << '\n';
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
      if (selfState_.haveMovingEventId()) {
        LOG(serverAgentEntityUpdatePositionReceived) << "Cancelling movement timer\n";
        eventBroker_.cancelDelayedEvent(selfState_.getMovingEventId());
        selfState_.resetMovingEventId();
      }
      selfState_.setPosition(packet.position());
      LOG(serverAgentEntityUpdatePositionReceived) << "Now stationary at " << selfState_.position() << '\n';
    } else {
      LOG(serverAgentEntityUpdatePositionReceived) << "We werent moving, weird\n";
      const auto pos = selfState_.position();
      LOG(serverAgentEntityUpdatePositionReceived) << "Expected pos: " << pos.xOffset << ',' << pos.zOffset << '\n';
      LOG(serverAgentEntityUpdatePositionReceived) << "Received pos: " << packet.position().xOffset << ',' << packet.position().zOffset << '\n';
      // TODO: Does it make sense to update our position in this case? Probably
      //  But it also seems like a problem because we mistakenly thought we were moving
      selfState_.setPosition(packet.position());
    }
  }
  return true;
}

bool MovementModule::clientAgentCharacterMoveRequestReceived(packet::parsing::ClientAgentCharacterMoveRequest &packet) {
  // Here, if we want to block the human from moving, we can return false
  return true;
}

float MovementModule::secondsToTravel(const packet::structures::Position &srcPosition, const packet::structures::Position &destPosition) const {
  auto distance = math::position::calculateDistance(srcPosition, destPosition);
  return distance / selfState_.currentSpeed();
}

bool MovementModule::serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) {
  if (packet.globalId() == selfState_.globalId()) {
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
    if (selfState_.haveMovingEventId()) {
      // Had a timer already running for movement, cancel it
      LOG(serverAgentEntityUpdateMovementReceived) << "Had a running timer, cancelling it" << std::endl;
      eventBroker_.cancelDelayedEvent(selfState_.getMovingEventId());
      selfState_.resetMovingEventId();
    }
    LOG(serverAgentEntityUpdateMovementReceived) << "We are moving from " << sourcePosition << ' ';
    if (packet.hasDestination()) {
      auto destPosition = packet.destinationPosition();
      std::cout << "to " << destPosition << '\n';
      if (sourcePosition.xOffset == destPosition.xOffset && sourcePosition.zOffset == destPosition.zOffset) {
        LOG(serverAgentEntityUpdateMovementReceived) << "Server says we're moving to our current position. wtf?\n";
        // Ignore this
      } else {
        auto seconds = secondsToTravel(sourcePosition, destPosition);
        LOG(serverAgentEntityUpdateMovementReceived) << "Should take " << seconds << "s. Timer set\n";
        const auto movingEventId = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
        selfState_.setMovingEventId(movingEventId);
        selfState_.setMoving(packet.destinationPosition());
      }
    } else {
      std::cout << "toward " << packet.angle() << '\n';
      selfState_.setMoving(packet.angle());
    }
  }
  return true;
}

bool MovementModule::clientAgentChatRequestReceived(packet::parsing::ClientAgentChatRequest &packet) {
  std::regex moveRegex(R"delim(move\s+(-?[0-9]+)\s+(-?[0-9]+))delim");
  std::regex move3Regex(R"delim(move\s+(-?[0-9]+)\s+(-?[0-9]+)\s+(-?[0-9]+))delim");
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
  }
  return true;
}

} // namespace module