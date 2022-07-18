#include "helpers.hpp"
#include "logging.hpp"
#include "packetProcessor.hpp"

PacketProcessor::PacketProcessor(state::Entity &entityState,
                                 state::Self &selfState,
                                 broker::PacketBroker &brokerSystem,
                                 broker::EventBroker &eventBroker,
                                 ui::UserInterface &userInterface,
                                 const packet::parsing::PacketParser &packetParser,
                                 const pk2::GameData &gameData) :
      entityState_(entityState),
      selfState_(selfState),
      broker_(brokerSystem),
      eventBroker_(eventBroker),
      userInterface_(userInterface),
      packetParser_(packetParser),
      gameData_(gameData) {
  subscribeToPackets();
}

void PacketProcessor::subscribeToPackets() {
  auto packetHandleFunction = std::bind(&PacketProcessor::handlePacket, this, std::placeholders::_1);

  // Client packets
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentAuthRequest, packetHandleFunction);

  // Server packets
  //   Login packets
  broker_.subscribeToServerPacket(packet::Opcode::LOGIN_SERVER_LIST, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::LOGIN_SERVER_AUTH_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::LOGIN_CLIENT_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_LOGIN_RESULT, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_CHARACTER, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_INGAME_ACCEPT, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerGatewayLoginIbuvChallenge, packetHandleFunction);
  // broker_.subscribeToServerPacket(static_cast<packet::Opcode>(0x6005), packetHandleFunction);
  //   Movement packets
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdateMovement, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntityUpdatePosition, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentEntitySyncPosition, packetHandleFunction);
}

bool PacketProcessor::handlePacket(const PacketContainer &packet) const {
  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    std::cerr << "[PacketProcessor] Failed to parse packet " << std::hex << packet.opcode << std::dec << "\n  Error: \"" << ex.what() << "\"\n";
    return true;
  }
  if (!parsedPacket) {
    // Not yet parsing this packet
    return true;
  }

  std::unique_lock<std::mutex> selfStateLock(selfState_.selfMutex);

  // Login packet handlers
  packet::parsing::ParsedLoginServerList *serverList = dynamic_cast<packet::parsing::ParsedLoginServerList*>(parsedPacket.get());
  if (serverList != nullptr) {
    serverListReceived(*serverList);
    return true;
  }

  packet::parsing::ParsedLoginResponse *loginResponse = dynamic_cast<packet::parsing::ParsedLoginResponse*>(parsedPacket.get());
  if (loginResponse != nullptr) {
    loginResponseReceived(*loginResponse);
    return true;
  }

  // This packet is a response to the client sending 0x2001 where the client indicates that it is the "SR_Client"
  packet::parsing::ParsedLoginClientInfo *loginClientInfo = dynamic_cast<packet::parsing::ParsedLoginClientInfo*>(parsedPacket.get());
  if (loginClientInfo != nullptr) {
    loginClientInfoReceived(*loginClientInfo);
    return true;
  }

  packet::parsing::ParsedUnknown *unknownPacket = dynamic_cast<packet::parsing::ParsedUnknown*>(parsedPacket.get());
  if (unknownPacket != nullptr) {
    auto fowardReturnValue = unknownPacketReceived(*unknownPacket);
    return fowardReturnValue;
  }

  packet::parsing::ParsedServerAuthResponse *serverAuthResponse = dynamic_cast<packet::parsing::ParsedServerAuthResponse*>(parsedPacket.get());
  if (serverAuthResponse != nullptr) {
    serverAuthReceived(*serverAuthResponse);
    return true;
  }

  packet::parsing::ParsedServerAgentCharacterSelectionActionResponse *charListResponse = dynamic_cast<packet::parsing::ParsedServerAgentCharacterSelectionActionResponse*>(parsedPacket.get());
  if (charListResponse != nullptr) {
    charListReceived(*charListResponse);
    return true;
  }

  packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse *charSelectionJoinResponse = dynamic_cast<packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse*>(parsedPacket.get());
  if (charSelectionJoinResponse != nullptr) {
    charSelectionJoinResponseReceived(*charSelectionJoinResponse);
    return true;
  }

  // Movement packet handlers
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

  LOG(handlePacket) << "Unhandled packet subscribed to\n";
  return true;
}

// ============================================================================================================================
// ===============================================Login process packet handling================================================
// ============================================================================================================================

void PacketProcessor::serverListReceived(const packet::parsing::ParsedLoginServerList &packet) const {
  selfState_.shardId = packet.shardId();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateShardIdUpdated));
}

void PacketProcessor::loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet) const {
  if (packet.result() == packet::enums::LoginResult::kSuccess) {
    selfState_.token = packet.token();
  } else {
    LOG(loginResponseReceived) << " Login failed\n";
  }
}

void PacketProcessor::loginClientInfoReceived(const packet::parsing::ParsedLoginClientInfo &packet) const {
  if (packet.serviceName() == "AgentServer") {
    // Connected to agentserver, send client auth packet
    selfState_.connectedToAgentServer = true;
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateConnectedToAgentServerUpdated));
  }
}

bool PacketProcessor::unknownPacketReceived(const packet::parsing::ParsedUnknown &packet) const {
  if (packet.opcode() == packet::Opcode::kClientAgentAuthRequest) {
    // The client is trying to authenticate
    if (selfState_.loggingIn) {
      // We're the ones logging in, the client is not in the correct state, so this packet will break the login process
      // Block this from the server
      return false;
    }
  } else if (packet.opcode() == packet::Opcode::kServerGatewayLoginIbuvChallenge) {
    // Got the captcha prompt, respond with an answer
    selfState_.receivedCaptchaPrompt = true;
    eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateReceivedCaptchaPromptUpdated));
  }
  return true;
}

void PacketProcessor::serverAuthReceived(const packet::parsing::ParsedServerAuthResponse &packet) const {
  if (packet.result() == 0x01) {
    // Successful login
    selfState_.loggingIn = false;
    // Client will automatically request the character listing
    // TODO: For clientless, we will need to do this ourself
  }
}

void PacketProcessor::charListReceived(const packet::parsing::ParsedServerAgentCharacterSelectionActionResponse &packet) const {
  selfState_.characterList = packet.characters();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateCharacterListUpdated));
}

void PacketProcessor::charSelectionJoinResponseReceived(const packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse &packet) const {
  // A character was selected after login, this is the response
  if (packet.result() != 0x01) {
    // Character selection failed
    // TODO: Properly handle error
    LOG(charSelectionJoinResponseReceived) << "Failed when selecting character\n";
  }
}

// ============================================================================================================================
// ==================================================Movement packet handling==================================================
// ============================================================================================================================

bool PacketProcessor::serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet) const {
  if (packet.globalId() == selfState_.globalId()) {
    const auto &currentPosition = selfState_.position();
    LOG(serverAgentEntitySyncPositionReceived) << "Syncing position: " << currentPosition.xOffset << ',' << currentPosition.zOffset << std::endl;
    selfState_.syncPosition(packet.position());
  }
  return false;
}

bool PacketProcessor::serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) const {
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

bool PacketProcessor::serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) const {
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
        auto seconds = secondsToTravel(sourcePosition, destPosition, selfState_.currentSpeed());
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