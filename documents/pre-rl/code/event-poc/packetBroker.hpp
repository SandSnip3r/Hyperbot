#ifndef PACKET_BROKER_HPP
#define PACKET_BROKER_HPP

class PacketBroker {
public:
  using SubscriptionId = uint32_t;
  SubscriptionId subscribe(Packet::Opcode opcode);
private:
};

#endif // PACKET_BROKER_HPP