#ifndef PACKET_BUILDING_SERVER_AGENT_INVENTORY_OPERATION_RESPONSE_HPP_
#define PACKET_BUILDING_SERVER_AGENT_INVENTORY_OPERATION_RESPONSE_HPP_

#include "packet/opcode.hpp"
#include "storage/item.hpp"

#include "../../shared/silkroad_security.h"

namespace packet::building {

class ServerAgentInventoryOperationResponse {
private:
  static const Opcode kOpcode_ = Opcode::kServerAgentInventoryOperationResponse;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer addItemByServerPacket(uint8_t targetSlot, const storage::Item &item);
};

} // namespace packet::building

#endif // PACKET_BUILDING_SERVER_AGENT_INVENTORY_OPERATION_RESPONSE_HPP_