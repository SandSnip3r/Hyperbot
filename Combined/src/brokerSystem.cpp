#include "brokerSystem.hpp"
#include <iostream>

void BrokerSystem::setInjectionFunction(PacketInjectionFunction &&injectionFunction) {
  injectionFunction_ = std::move(injectionFunction);
}

bool BrokerSystem::packetReceived(const PacketContainer &packet, PacketContainer::Direction packetDirection) {
  //COUT std::cout << "Broker received packet " << PacketPrinting::opcodeToStr(packet.opcode()) << "\n";
  // A new packet has arrived
  // First, determine which "event bus" to "put it on"
  PacketSubscriptionMap *subscriptionMap;
  if (packetDirection == PacketContainer::Direction::kClientToServer) {
    // The packet goes on the Client->Server event bus
    subscriptionMap = &clientPacketSubscriptions_;
  } else if (packetDirection == PacketContainer::Direction::kServerToClient) {
    // The packet goes on the Server->Client event bus
    subscriptionMap = &serverPacketSubscriptions_;
  } else {
    return true;
  }
  
  
  bool forwardPacket=true;
  // Check if anybody is subscribed to this packet
  auto subscriptionIt = subscriptionMap->find(static_cast<Opcode>(packet.opcode));
  if (subscriptionIt != subscriptionMap->end()) {
    // We have one or more subscribers for this opcode, forward it to each
    std::vector<PacketHandleFunction> &handleFunctions = subscriptionIt->second;
    for (auto &handleFunction : handleFunctions) {
      // Wrap the packet into an object with lazy-eval-parsing
      std::unique_ptr<PacketParsing::PacketParser> packetParser{PacketParsing::newPacketParser(packet)};
      // Send this packet and make note if the packet should be forwarded
      //COUT std::cout << " Broker sending packet " << PacketPrinting::opcodeToStr(packet.opcode()) << " to subscriber\n";
      forwardPacket &= handleFunction(packetParser);
    }
  } else {
    //COUT std::cout << " Nobody subscribed\n";
  }
  return forwardPacket;
}

void BrokerSystem::injectPacket(const PacketContainer &packet, const PacketContainer::Direction packetDirection) {
  injectionFunction_(packet, packetDirection);
}

void BrokerSystem::subscribeToClientPacket(Opcode opcode, PacketHandleFunction &&handleFunc) {
  subscribeToPacket(clientPacketSubscriptions_, opcode, std::move(handleFunc));
}

void BrokerSystem::subscribeToServerPacket(Opcode opcode, PacketHandleFunction &&handleFunc) {
  subscribeToPacket(serverPacketSubscriptions_, opcode, std::move(handleFunc));
}

void BrokerSystem::subscribeToPacket(PacketSubscriptionMap &subscriptions, Opcode opcode, PacketHandleFunction &&handleFunc) {
  auto subscriptionIt = subscriptions.find(opcode);
  if (subscriptionIt == subscriptions.end()) {
    auto itBoolResult = subscriptions.emplace(opcode, std::vector<PacketHandleFunction>());
    if (!itBoolResult.second) {
      std::cerr << "Unable to subscribe!\n";
      // TODO: Handle error better
      return;
    } else {
      subscriptionIt = itBoolResult.first;
    }
  }
  std::cout << "Successfully subscribed to " << std::hex << static_cast<uint16_t>(subscriptionIt->first) << std::dec << '\n';
  subscriptionIt->second.emplace_back(std::move(handleFunc));
  std::cout << " -clientPacketSubscriptions.size() = " << clientPacketSubscriptions_.size() << '\n';
  std::cout << " -serverPacketSubscriptions.size() = " << serverPacketSubscriptions_.size() << '\n';
}