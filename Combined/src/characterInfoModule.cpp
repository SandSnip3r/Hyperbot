#include "characterInfoModule.hpp"
#include "opcode.hpp"
#include "packetBuilding.hpp"

#include <iostream>
#include <memory>

CharacterInfoModule::CharacterInfoModule(BrokerSystem &brokerSystem,
                         const packet::parsing::PacketParser &packetParser) :
      broker_(brokerSystem),
      packetParser_(packetParser) {
  auto packetHandleFunction = std::bind(&CharacterInfoModule::handlePacket, this, std::placeholders::_1);
  // Client packets
  // Server packets
  broker_.subscribeToServerPacket(Opcode::SERVER_CHARDATA, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::SERVER_HPMP_UPDATE, packetHandleFunction);
}

bool CharacterInfoModule::handlePacket(const PacketContainer &packet) {
  std::cout << "CharacterInfoModule::handlePacket\n";

  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    std::cerr << "[CharacterInfoModule] Failed to parse packet " << std::hex << packet.opcode << std::dec << "\n  Error: \"" << ex.what() << "\"\n";
    return true;
  }

  if (!parsedPacket) {
    // Not yet parsing this packet
    return true;
  }

  packet::parsing::ParsedServerAgentCharacterData *charData = dynamic_cast<packet::parsing::ParsedServerAgentCharacterData*>(parsedPacket.get());
  if (charData != nullptr) {
    characterInfoReceived(*charData);
    return true;
  }

  packet::parsing::ParsedServerHpMpUpdate *hpMpUpdate = dynamic_cast<packet::parsing::ParsedServerHpMpUpdate*>(parsedPacket.get());
  if (hpMpUpdate != nullptr) {
    return true;
  }

  std::cout << "Unhandled packet subscribed to\n";
  return true;
}

void CharacterInfoModule::characterInfoReceived(const packet::parsing::ParsedServerAgentCharacterData &packet) {
  std::cout << "Character data received\n";
}