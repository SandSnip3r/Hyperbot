#include "../opcode.hpp"
#include "../../shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_AGENT_INVENTORY_ITEM_USE_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_INVENTORY_ITEM_USE_REQUEST_HPP

namespace packet::building {

class ClientAgentInventoryItemUseRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentInventoryItemUseRequest;
  static const bool kEncrypted_ = true;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint8_t slotNum, uint16_t itemData);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_INVENTORY_ITEM_USE_REQUEST_HPP