#include "opcode.hpp"
#include "packetParsing.hpp"
#include "shared/silkroad_security.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#ifndef BROKER_SYSTEM_HPP
#define BROKER_SYSTEM_HPP

/// BrokerSystem
/// TODO: Maybe the PacketHandleFunctions should take a polymorphic type which is a wrapper around the packet that provides lazy evaluation for the data that's parsed
/// TODO: Unsubscription from opcodes
/// TODO: Allow registration of multiple events in one call if necessary
class BrokerSystem {
private:
  using PacketHandleFunction = std::function<bool(std::unique_ptr<PacketParsing::PacketParser>&)>;
  using PacketInjectionFunction = std::function<void(const PacketContainer&, const PacketContainer::Direction)>;
  using PacketSubscriptionMap = std::unordered_map<Opcode, std::vector<PacketHandleFunction>>;
  PacketSubscriptionMap clientPacketSubscriptions_, serverPacketSubscriptions_;
  PacketInjectionFunction injectionFunction_;
  void subscribeToPacket(PacketSubscriptionMap &subscriptions, Opcode opcode, PacketHandleFunction &&handleFunc);
public:
  void setInjectionFunction(PacketInjectionFunction &&injectionFunction);
  bool packetReceived(const PacketContainer &packet, PacketContainer::Direction packetDirection);
  void injectPacket(const PacketContainer &packet, const PacketContainer::Direction packetDirection);
  void subscribeToClientPacket(Opcode opcode, PacketHandleFunction &&handleFunc);
  void subscribeToServerPacket(Opcode opcode, PacketHandleFunction &&handleFunc);
};

#endif // BROKER_SYSTEM_HPP