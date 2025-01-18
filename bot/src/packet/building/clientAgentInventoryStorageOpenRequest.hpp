#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_INVENTORY_STORAGE_OPEN_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_INVENTORY_STORAGE_OPEN_REQUEST_HPP

namespace packet::building {

class ClientAgentInventoryStorageOpenRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentInventoryStorageOpenRequest;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint32_t gId);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_INVENTORY_STORAGE_OPEN_REQUEST_HPP