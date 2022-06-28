#include "../opcode.hpp"
#include "../../shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_INVENTORY_OPERATION_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_INVENTORY_OPERATION_REQUEST_HPP

namespace packet::building {

class ClientAgentInventoryOperationRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentInventoryOperationRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity);
  static PacketContainer packet(uint64_t goldDropAmount);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_INVENTORY_OPERATION_REQUEST_HPP