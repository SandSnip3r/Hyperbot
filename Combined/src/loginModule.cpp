#include "loginModule.hpp"
#include "opcode.hpp"
#include "packetBuilding.hpp"

#include <iostream>
#include <Windows.h>

LoginModule::LoginModule(const config::ConfigData &configData, BrokerSystem &brokerSystem) :
      broker_(brokerSystem),
      kCharName_(configData.charName()),
      kUsername_(configData.charId()),
      kPassword_(configData.charPassword()) {
  // Client packets
  broker_.subscribeToClientPacket(Opcode::CLIENT_CAFE, std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
  broker_.subscribeToClientPacket(Opcode::CLIENT_AUTH, std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
  // Server packets
  broker_.subscribeToServerPacket(Opcode::LOGIN_SERVER_LIST, std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
  broker_.subscribeToServerPacket(Opcode::LOGIN_SERVER_AUTH_INFO, std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
  broker_.subscribeToServerPacket(Opcode::LOGIN_CLIENT_INFO, std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
  broker_.subscribeToServerPacket(Opcode::SERVER_LOGIN_RESULT, std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
  broker_.subscribeToServerPacket(Opcode::SERVER_CHARACTER, std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
  broker_.subscribeToServerPacket(Opcode::SERVER_INGAME_ACCEPT, std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
  broker_.subscribeToServerPacket(Opcode::SERVER_CHARDATA, std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
  // broker_.subscribeToServerPacket(static_cast<Opcode>(0x6005), std::bind(&LoginModule::handlePacket, this, std::placeholders::_1));
}

bool LoginModule::handlePacket(std::unique_ptr<PacketParsing::PacketParser> &packetParser) {
  std::cout << "LoginModule::handlePacket\n";
  if (!packetParser) {
    // No packet parser
    return true;
  }

  PacketParsing::ClientCafePacket *cafe = dynamic_cast<PacketParsing::ClientCafePacket*>(packetParser.get());
  if (cafe != nullptr) {
    cafeReceived();
    return true;
  }

  PacketParsing::LoginServerListPacket *serverList = dynamic_cast<PacketParsing::LoginServerListPacket*>(packetParser.get());
  if (serverList != nullptr) {
    serverListReceived(*serverList);
    return true;
  }

  PacketParsing::LoginResponsePacket *loginResponse = dynamic_cast<PacketParsing::LoginResponsePacket*>(packetParser.get());
  if (loginResponse != nullptr) {
    loginResponseReceived(*loginResponse);
    return true;
  }

  // This packet is a response to the client sending 0x2001 where the client indicates that it is the "SR_Client"
  PacketParsing::LoginClientInfoPacket *loginClientInfo = dynamic_cast<PacketParsing::LoginClientInfoPacket*>(packetParser.get());
  if (loginClientInfo != nullptr) {
    loginClientInfoReceived(*loginClientInfo);
    return true;
  }

  PacketParsing::UnknownPacket *unknownPacket = dynamic_cast<PacketParsing::UnknownPacket*>(packetParser.get());
  if (unknownPacket != nullptr) {
    auto fowardReturnValue = unknownPacketReceived(*unknownPacket);
    return fowardReturnValue;
  }

  PacketParsing::ServerAuthResponsePacket *serverAuthResponse = dynamic_cast<PacketParsing::ServerAuthResponsePacket*>(packetParser.get());
  if (serverAuthResponse != nullptr) {
    serverAuthReceived(*serverAuthResponse);
    return true;
  }

  PacketParsing::ServerAgentCharacterSelectionActionResponsePacket *charListResponse = dynamic_cast<PacketParsing::ServerAgentCharacterSelectionActionResponsePacket*>(packetParser.get());
  if (charListResponse != nullptr) {
    charListReceived(*charListResponse);
    return true;
  }

  PacketParsing::ServerAgentCharacterSelectionJoinResponsePacket *charSelectionJoinResponse = dynamic_cast<PacketParsing::ServerAgentCharacterSelectionJoinResponsePacket*>(packetParser.get());
  if (charSelectionJoinResponse != nullptr) {
    charSelectionJoinResponseReceived(*charSelectionJoinResponse);
    return true;
  }

  PacketParsing::ServerAgentCharacterDataPacket *charData = dynamic_cast<PacketParsing::ServerAgentCharacterDataPacket*>(packetParser.get());
  if (charData != nullptr) {
    std::cout << "Got character data\n";
    return true;
  }

  std::cout << "Unhandled packet subscribed to\n";
  return true;
}

void LoginModule::cafeReceived() {
  std::cout << " CAFE packet received, injecting loginauth packet\n";
  auto loginAuthPacket = PacketBuilding::LoginAuthPacketBuilder(kLocale_, kUsername_, kPassword_, shardId_).packet();
  broker_.injectPacket(loginAuthPacket, PacketContainer::Direction::kClientToServer);
}

void LoginModule::serverListReceived(PacketParsing::LoginServerListPacket &packet) {
  std::cout << "Server List Received\n";
  shardId_ = packet.shardId();
  std::cout << " Server List packet, shardId:" << shardId_ << "\n";
}

void LoginModule::loginResponseReceived(PacketParsing::LoginResponsePacket &packet) {
  std::cout << " Login response, result:" << static_cast<int>(packet.result()) << ", token:" << packet.token() << "\n";
  if (packet.result() != PacketEnums::LoginResult::kSuccess) {
    std::cout << " Login failed\n";
  } else {
    std::cout << " Login Success, saving token\n";
    token_ = packet.token();
  }
}

void LoginModule::loginClientInfoReceived(PacketParsing::LoginClientInfoPacket &packet) {
  std::cout << " Login client info, service name:" << packet.serviceName() << "\n";
  if (packet.serviceName() != "AgentServer") {
    std::cout << "Not agentserver\n";
  } else {
    std::cout << "Injecting client auth packet to agentserver\n";
    // Connected to agentserver, send client auth packet
    auto clientAuthPacket = PacketBuilding::ClientAuthPacketBuilder(token_, kUsername_, kPassword_, kLocale_, kMacAddress_).packet();
    broker_.injectPacket(clientAuthPacket, PacketContainer::Direction::kClientToServer);
    loggingIn_ = true;
    // Allow this packet to continue to the client
    // Warning: The client will send an empty clientAuth packet. block it from the server
  }
}

bool LoginModule::unknownPacketReceived(PacketParsing::UnknownPacket &packet) {
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

void LoginModule::serverAuthReceived(PacketParsing::ServerAuthResponsePacket &packet) {
  std::cout << " Server auth response: " << (int)packet.result() << "\n";
  if (packet.result() == 0x01) {
    loggingIn_ = false;
    // TODO: remove this function and opcode subscription. Client will already take care of this
    // std::cout << "Successfully logged in! Request character list\n";
    // auto characterListPacket = PacketBuilding::CharacterSelectionActionPacketBuilder(PacketEnums::CharacterSelectionAction::kList).packet();
    // broker_.injectPacket(characterListPacket, PacketContainer::Direction::kClientToServer);
  }
}

void LoginModule::charListReceived(PacketParsing::ServerAgentCharacterSelectionActionResponsePacket &packet) {
  auto &charList = packet.characters();
  std::cout << "Char list received, " << charList.size() << " character(s)\n";
  // Search for our character in the character list
  auto it = std::find_if(charList.begin(), charList.end(), [this](const PacketInnerStructures::CharacterSelection::Character &character) {
    return character.name == kCharName_;
  });
  if (it == charList.end()) {
    std::cout << "Unable to find character \"" << kCharName_ << "\". Options are [";
    for (const auto &character : charList) {
      std::cout << character.name << ',';
    }
    std::cout << "]\n";
    return;
  }
  // Found our character, select it
  auto charSelectionPacket = PacketBuilding::ClientAgentSelectionJoinPacketBuilder(kCharName_).packet();
  broker_.injectPacket(charSelectionPacket, PacketContainer::Direction::kClientToServer);
}

void LoginModule::charSelectionJoinResponseReceived(PacketParsing::ServerAgentCharacterSelectionJoinResponsePacket &packet) {
  // A character was selected after login, this is the response
  if (packet.result() != 0x01) {
    // Character selection failed
    // TODO: Properly handle error
    std::cout << "Failed when selecting character\n";
  }
}