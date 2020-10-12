#include "movementModule.hpp"
#include "../math/position.hpp"
#include "../packet/opcode.hpp"
#include "../packet/building/clientAgentCharacterMoveRequest.hpp"

#include <array>
#include <numeric>
#include <iostream>
#include <memory>
#include <regex>

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
  // TODO: When teleporting, cancel movement timer!!

  // event::EventCode::kMovementEnded
  auto eventHandleFunction = std::bind(&MovementModule::handleEvent, this, std::placeholders::_1);
  eventBroker_.subscribeToEvent(event::EventCode::kMovementEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterSpeedUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kTemp, eventHandleFunction);

  eng_ = createRandomEngine();
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
    default:
      std::cout << "Unhandled event subscribed to. Code:" << static_cast<int>(eventCode) << '\n';
      break;
  }
}

void MovementModule::handleTempEvent() {
}

void MovementModule::handleSpeedUpdated() {
  if (movingEventId_ && selfState_.moving() && selfState_.haveDestination()) {
    // Need to update timer
    eventBroker_.cancelDelayedEvent(*movingEventId_);
    movingEventId_.reset();
    auto seconds = secondsToTravel(selfState_.position(), selfState_.destination());
    movingEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
  }
}

void MovementModule::handleMovementEnded() {
  movingEventId_.reset();
  std::cout << "[TIMER END] Movement ended event\n";
  selfState_.doneMoving();
  auto currentPos = selfState_.position();
  std::cout << "[TIMER END] Currently at " << selfState_.position() << '\n';
  if (movingErratically_) {
    std::uniform_int_distribution<int> xDist(-maxXOffset_,maxXOffset_);
    std::uniform_int_distribution<int> zDist(-maxZOffset_,maxZOffset_);
    auto newPos = math::position::offset(center_, xDist(eng_), zDist(eng_));
    broker_.injectPacket(packet::building::ClientAgentCharacterMoveRequest::packet(newPos.regionId, newPos.xOffset, newPos.yOffset, newPos.zOffset), PacketContainer::Direction::kClientToServer);
  }
}

bool MovementModule::serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) {
  if (packet.globalId() == selfState_.globalId()) {
    if (selfState_.moving()) {
      std::cout << "[UPDATE POS] We were moving. Must've hit an obstacle\n";
      if (movingEventId_) {
        std::cout << "[UPDATE POS] Cancelling timer\n";
        eventBroker_.cancelDelayedEvent(*movingEventId_);
        movingEventId_.reset();
      }
      selfState_.setPosition(packet.position());
      std::cout << "[UPDATE POS] Now at " << selfState_.position() << '\n';
    } else {
      std::cout << "[UPDATE POS] We werent moving, weird\n";
    }
  }
  return true;
}

bool MovementModule::clientAgentCharacterMoveRequestReceived(packet::parsing::ClientAgentCharacterMoveRequest &packet) {
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
  std::cout << "secondsToTravel distance = " << distance << '\n';
  return distance / selfState_.currentSpeed();
}

bool MovementModule::serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) {
  if (packet.globalId() == selfState_.globalId()) {
    packet::structures::Position sourcePosition;
    if (packet.hasSource()) {
      // Use given position
      sourcePosition = packet.sourcePosition();
    } else {
      // Use known position
      sourcePosition = selfState_.position();
    }
    if (movingEventId_) {
      // Had a timer already running for movement, cancel it
      std::cout << "[UPDATE MOVEMENT] Had a running timer, cancelling it\n";
      eventBroker_.cancelDelayedEvent(*movingEventId_);
      movingEventId_.reset();
    }
    std::cout << "[UPDATE MOVEMENT] We are moving from " << sourcePosition << ' ';
    if (packet.hasDestination()) {
      auto destPosition = packet.destinationPosition();
      auto seconds = secondsToTravel(sourcePosition, destPosition);
      std::cout << "to " << destPosition << '\n';
      std::cout << "[UPDATE MOVEMENT]   Should take " << seconds << "s. Timer set\n";
      movingEventId_ = eventBroker_.publishDelayedEvent(std::make_unique<event::Event>(event::EventCode::kMovementEnded), std::chrono::milliseconds(static_cast<uint64_t>(seconds*1000)));
      selfState_.setMoving(packet.destinationPosition());
    } else {
      std::cout << "toward " << packet.angle() << '\n';
      selfState_.setMoving(packet.angle());
    }
    // TODO: Emit event "character began moving"
  }
  return true;
}

void MovementModule::startMovingErratically() {
  movingErratically_ = true;
  center_ = selfState_.position();
  std::uniform_int_distribution<int> xDist(-maxXOffset_,maxXOffset_);
  std::uniform_int_distribution<int> zDist(-maxZOffset_,maxZOffset_);
  auto newPos = math::position::offset(center_, xDist(eng_), zDist(eng_));
  broker_.injectPacket(packet::building::ClientAgentCharacterMoveRequest::packet(newPos.regionId, newPos.xOffset, newPos.yOffset, newPos.zOffset), PacketContainer::Direction::kClientToServer);
}

void MovementModule::stopMovingErratically() {
  movingErratically_ = false;
}

bool MovementModule::clientAgentChatRequestReceived(packet::parsing::ClientAgentChatRequest &packet) {
  std::regex startRegex(R"delim(start ([0-9]+))delim");
  // std::regex startRegex(R"delim(start ([0-9]+) (true|false))delim");
  std::regex stopRegex(R"delim(stop)delim");
  std::regex moveRegex(R"delim(move (-?[0-9]+) (-?[0-9]+))delim");
  std::smatch regexMatch;
  if (std::regex_match(packet.message(), regexMatch, startRegex)) {
    //start
    maxXOffset_ = std::stoi(regexMatch[1].str());
    maxZOffset_ = std::stoi(regexMatch[1].str());
    startMovingErratically();
  } else if (std::regex_match(packet.message(), regexMatch, stopRegex)) {
    //stop
    stopMovingErratically();
  } else if (std::regex_match(packet.message(), regexMatch, moveRegex)) {
    const int x = std::stoi(regexMatch[1].str());
    const int y = std::stoi(regexMatch[2].str());
    // Move character
    std::cout << "Moving character to " << x << ',' << y << '\n';
    // | 15 | 14 | 13 | 12 | 11 | 10 | 09 | 08 | 07 | 06 | 05 | 04 | 03 | 02 | 01 | 00 |
    // | DF |            ZSector               |                XSector                |
    // DF = Dungeon flag
    // 6B 69                                             ki..............
    // 76 07                                             v...............
    // B4 00                                             ................
    // 98 06  
    // -5186,2664
    // 0b0 1101001 01101011
    // No dungeon, 0x69 z sector, 0x6b x sector
    // X offset 0x767
    // Y offset 0xB4
    // Z offset 0x698
    struct XY {
      int x,y;
    };
    struct RegionXYZ {
      int regionId;
      int x,y,z;
    };
    auto converterFunc = [](const XY &obj) {
      RegionXYZ result;
      // byte ySector = (byte)(Region >> 8);
      // byte xSector = (byte)(Region & 0xFF);

      // if (Region < short.MaxValue) // At world map? convert it!
      // {
      //     // 192px 1:10 scale, center (135,92)
      //     PosX = (xSector - 135) * 192 + X / 10;
      //     PosY = (ySector - 92) * 192 + Y / 10;
      // } 
      // result.regionId
    };
    // broker_.injectPacket(packet::building::ClientAgentCharacterMoveRequest::packet(), PacketContainer::Direction::kClientToServer);
    return false;
  } else {
    return true;
  }
}

// void MovementModule::serverAgentActionSelectResponseReceived(packet::parsing::ServerAgentActionSelectResponse &packet) {}

} // namespace module