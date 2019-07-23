#include "loginModule.hpp"
#include "opcode.hpp"
#include "packetBuilding.hpp"

#include <iostream>
#include <Windows.h>

LoginModule::LoginModule(BrokerSystem &brokerSystem,
                         const packet::parsing::PacketParser &packetParser,
                         const config::CharacterLoginData &loginData,
                         const pk2::DivisionInfo &divisionInfo) :
      broker_(brokerSystem),
      packetParser_(packetParser),
      loginData_(loginData),
      divisionInfo_(divisionInfo) {
  auto packetHandleFunction = std::bind(&LoginModule::handlePacket, this, std::placeholders::_1);
  // Client packets
  broker_.subscribeToClientPacket(Opcode::CLIENT_CAFE, packetHandleFunction);
  broker_.subscribeToClientPacket(Opcode::CLIENT_AUTH, packetHandleFunction);
  // Server packets
  broker_.subscribeToServerPacket(Opcode::LOGIN_SERVER_LIST, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::LOGIN_SERVER_AUTH_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::LOGIN_CLIENT_INFO, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::SERVER_LOGIN_RESULT, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::SERVER_CHARACTER, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::SERVER_INGAME_ACCEPT, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::SERVER_CHARDATA, packetHandleFunction);
  // broker_.subscribeToServerPacket(static_cast<Opcode>(0x6005), packetHandleFunction);
}

bool LoginModule::handlePacket(const PacketContainer &packet) {
  std::cout << "LoginModule::handlePacket\n";

  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket = packetParser_.parsePacket(packet);
  if (!parsedPacket) {
    // Not yet parsing this packet
    return true;
  }

  packet::parsing::ParsedClientCafe *cafe = dynamic_cast<packet::parsing::ParsedClientCafe*>(parsedPacket.get());
  if (cafe != nullptr) {
    cafeReceived();
    return true;
  }

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

  packet::parsing::ParsedServerAgentCharacterData *charData = dynamic_cast<packet::parsing::ParsedServerAgentCharacterData*>(parsedPacket.get());
  if (charData != nullptr) {
    std::cout << "Got character data\n";
    return true;
  }

  std::cout << "Unhandled packet subscribed to\n";
  return true;
}

void LoginModule::cafeReceived() {
  std::cout << " CAFE packet received, injecting loginauth packet\n";
  auto loginAuthPacket = PacketBuilding::LoginAuthPacketBuilder(divisionInfo_.locale, loginData_.id, loginData_.password, shardId_).packet();
  broker_.injectPacket(loginAuthPacket, PacketContainer::Direction::kClientToServer);
}

void LoginModule::serverListReceived(const packet::parsing::ParsedLoginServerList &packet) {
  std::cout << "Server List Received\n";
  shardId_ = packet.shardId();
  std::cout << " Server List packet, shardId:" << shardId_ << "\n";
}

void LoginModule::loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet) {
  std::cout << " Login response, result:" << static_cast<int>(packet.result()) << ", token:" << packet.token() << "\n";
  if (packet.result() != PacketEnums::LoginResult::kSuccess) {
    std::cout << " Login failed\n";
  } else {
    std::cout << " Login Success, saving token\n";
    token_ = packet.token();
  }
}

void LoginModule::loginClientInfoReceived(const packet::parsing::ParsedLoginClientInfo &packet) {
  std::cout << " Login client info, service name:" << packet.serviceName() << "\n";
  if (packet.serviceName() != "AgentServer") {
    std::cout << "Not agentserver\n";
  } else {
    std::cout << "Injecting client auth packet to agentserver\n";
    // Connected to agentserver, send client auth packet
    auto clientAuthPacket = PacketBuilding::ClientAuthPacketBuilder(token_, loginData_.id, loginData_.password, divisionInfo_.locale, kMacAddress_).packet();
    broker_.injectPacket(clientAuthPacket, PacketContainer::Direction::kClientToServer);
    loggingIn_ = true;
    // Allow this packet to continue to the client
    // Warning: The client will send an empty clientAuth packet. block it from the server
  }
}

bool LoginModule::unknownPacketReceived(const packet::parsing::ParsedUnknown &packet) {
  std::cout << "Unknown packet\n";
  if (packet.opcode() == Opcode::CLIENT_AUTH) {
    std::cout << "Client_auth packet, actually\n";
    // Client auth packet
    if (loggingIn_) {
      std::cout << "client trying to authenticate\n";
      // Block this from going to the server
      return false;
    }
  }
  return true;
}

void LoginModule::serverAuthReceived(const packet::parsing::ParsedServerAuthResponse &packet) {
  std::cout << " Server auth response: " << (int)packet.result() << "\n";
  if (packet.result() == 0x01) {
    loggingIn_ = false;
    // TODO: remove this function and opcode subscription. Client will already take care of this
    // std::cout << "Successfully logged in! Request character list\n";
    // auto characterListPacket = PacketBuilding::CharacterSelectionActionPacketBuilder(PacketEnums::CharacterSelectionAction::kList).packet();
    // broker_.injectPacket(characterListPacket, PacketContainer::Direction::kClientToServer);
  }
}

void LoginModule::charListReceived(const packet::parsing::ParsedServerAgentCharacterSelectionActionResponse &packet) {
  auto &charList = packet.characters();
  std::cout << "Char list received, " << charList.size() << " character(s)\n";
  // Search for our character in the character list
  auto it = std::find_if(charList.begin(), charList.end(), [this](const PacketInnerStructures::CharacterSelection::Character &character) {
    return character.name == loginData_.name;
  });
  if (it == charList.end()) {
    std::cout << "Unable to find character \"" << loginData_.name << "\". Options are [";
    for (const auto &character : charList) {
      std::cout << character.name << ',';
    }
    std::cout << "]\n";
    return;
  }
  // Found our character, select it
  auto charSelectionPacket = PacketBuilding::ClientAgentSelectionJoinPacketBuilder(loginData_.name).packet();
  broker_.injectPacket(charSelectionPacket, PacketContainer::Direction::kClientToServer);
}

void LoginModule::charSelectionJoinResponseReceived(const packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse &packet) {
  // A character was selected after login, this is the response
  if (packet.result() != 0x01) {
    // Character selection failed
    // TODO: Properly handle error
    std::cout << "Failed when selecting character\n";
  }
}