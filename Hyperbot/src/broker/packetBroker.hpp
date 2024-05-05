#ifndef BROKER_SYSTEM_HPP
#define BROKER_SYSTEM_HPP

#include "../packet/opcode.hpp"
#include "../shared/silkroad_security.h"

#include <functional>
#include <unordered_map>
#include <vector>

namespace broker {

/// PacketBroker
/// Does not have its own thread.
///   Proxy's thread calls packetReceived.
///   Proxy's thread or Bot's thread calls injectPacket (ideally only Bot's thread)
/// TODO: Maybe the PacketHandleFunctions should take a polymorphic type which is a wrapper around the packet that provides lazy evaluation for the data that's parsed
/// TODO: Unsubscription from opcodes
/// TODO: Allow registration of multiple events in one call if necessary
class PacketBroker {
private:
  using PacketHandleFunction = std::function<void(const PacketContainer&)>;
  using PacketInjectionFunction = std::function<void(const PacketContainer&, const PacketContainer::Direction)>;
  using PacketSubscriptionMap = std::unordered_map<packet::Opcode, std::vector<PacketHandleFunction>>;
  PacketSubscriptionMap clientPacketSubscriptions_, serverPacketSubscriptions_;
  PacketInjectionFunction injectionFunction_;
  void subscribeToPacket(PacketSubscriptionMap &subscriptions, packet::Opcode opcode, PacketHandleFunction &&handleFunc);
public:
  void setInjectionFunction(PacketInjectionFunction &&injectionFunction);
  void packetReceived(const PacketContainer &packet, PacketContainer::Direction packetDirection);
  void injectPacket(const PacketContainer &packet, const PacketContainer::Direction packetDirection);
  void subscribeToClientPacket(packet::Opcode opcode, PacketHandleFunction &&handleFunc);
  void subscribeToServerPacket(packet::Opcode opcode, PacketHandleFunction &&handleFunc);
};

} // namespace broker

#endif // BROKER_SYSTEM_HPP