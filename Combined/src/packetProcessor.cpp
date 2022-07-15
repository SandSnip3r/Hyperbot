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
  broker_.subscribeToServerPacket(packet::Opcode::LOGIN_SERVER_LIST, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::LOGIN_SERVER_AUTH_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::LOGIN_CLIENT_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_LOGIN_RESULT, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_CHARACTER, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::SERVER_INGAME_ACCEPT, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerGatewayLoginIbuvChallenge, packetHandleFunction);
  // broker_.subscribeToServerPacket(static_cast<packet::Opcode>(0x6005), packetHandleFunction);
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

  std::cout << "Unhandled packet subscribed to\n";
  return true;
}

void PacketProcessor::serverListReceived(const packet::parsing::ParsedLoginServerList &packet) const {
  selfState_.shardId = packet.shardId();
  eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStateShardIdUpdated));
}

void PacketProcessor::loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet) const {
  if (packet.result() == packet::enums::LoginResult::kSuccess) {
    selfState_.token = packet.token();
  } else {
    std::cout << " Login failed\n";
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
    std::cout << "Failed when selecting character\n";
  }
}