#include "type_id/typeCategory.hpp"

#include "../opcode.hpp"
#include "../../shared/silkroad_security.h"

#include <silkroad_lib/scalar_types.h>

#ifndef PACKET_BUILDING_CLIENT_AGENT_INVENTORY_ITEM_USE_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_INVENTORY_ITEM_USE_REQUEST_HPP

namespace packet::building {

class ClientAgentInventoryItemUseRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentInventoryItemUseRequest;
  static const bool kEncrypted_ = true;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(sro::scalar_types::StorageIndexType inventoryIndex, type_id::TypeId typeId);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_INVENTORY_ITEM_USE_REQUEST_HPP