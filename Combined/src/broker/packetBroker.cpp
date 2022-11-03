#include "logging.hpp"
#include "packetBroker.hpp"

#include <stdexcept>

namespace broker {

void PacketBroker::setInjectionFunction(PacketInjectionFunction &&injectionFunction) {
  injectionFunction_ = std::move(injectionFunction);
}

void PacketBroker::packetReceived(const PacketContainer &packet, PacketContainer::Direction packetDirection) {
  // A new packet has arrived
  // First, determine which "event bus" to "put it on"
  PacketSubscriptionMap *subscriptionMap;
  if (packetDirection == PacketContainer::Direction::kClientToServer || packetDirection == PacketContainer::Direction::kBotToServer) {
    // The packet goes on the Client->Server event bus
    subscriptionMap = &clientPacketSubscriptions_;
  } else if (packetDirection == PacketContainer::Direction::kServerToClient) {
    // The packet goes on the Server->Client event bus
    subscriptionMap = &serverPacketSubscriptions_;
  } else {
    // PacketContainer::Direction::kBotToClient
    //  Not handling this case, as it's usually to spoof something in the client
    return;
  }
  
  
  // Check if anybody is subscribed to this packet
  auto subscriptionIt = subscriptionMap->find(static_cast<packet::Opcode>(packet.opcode));
  if (subscriptionIt != subscriptionMap->end()) {
    // We have one or more subscribers for this opcode, forward it to each
    std::vector<PacketHandleFunction> &handleFunctions = subscriptionIt->second;
    for (auto &handleFunction : handleFunctions) {
      // Send this packet and make note if the packet should be forwarded
      handleFunction(packet);
    }
  }
}

void PacketBroker::injectPacket(const PacketContainer &packet, const PacketContainer::Direction packetDirection) {
  injectionFunction_(packet, packetDirection);
}

void PacketBroker::subscribeToClientPacket(packet::Opcode opcode, PacketHandleFunction &&handleFunc) {
  subscribeToPacket(clientPacketSubscriptions_, opcode, std::move(handleFunc));
}

void PacketBroker::subscribeToServerPacket(packet::Opcode opcode, PacketHandleFunction &&handleFunc) {
  subscribeToPacket(serverPacketSubscriptions_, opcode, std::move(handleFunc));
}

void PacketBroker::subscribeToPacket(PacketSubscriptionMap &subscriptions, packet::Opcode opcode, PacketHandleFunction &&handleFunc) {
  auto subscriptionIt = subscriptions.find(opcode);
  if (subscriptionIt == subscriptions.end()) {
    auto itBoolResult = subscriptions.emplace(opcode, std::vector<PacketHandleFunction>());
    if (!itBoolResult.second) {
      throw std::runtime_error("PacketBroker unable to create subscription for opcode "+std::to_string(int(opcode)));
    } else {
      subscriptionIt = itBoolResult.first;
    }
  }
  // TODO: Handle accidental duplicate subscriptions
  subscriptionIt->second.emplace_back(std::move(handleFunc));
}

} // namespace broker