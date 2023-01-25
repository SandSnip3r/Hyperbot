#ifndef PACKET_PARSING_CLIENT_AGENT_INVENTORY_ITEM_USE_REQUEST_HPP_
#define PACKET_PARSING_CLIENT_AGENT_INVENTORY_ITEM_USE_REQUEST_HPP_

#include "packet/parsing/parsedPacket.hpp"
#include "type_id/typeCategory.hpp"

#include <silkroad_lib/scalar_types.h>

namespace packet::parsing {

class ClientAgentInventoryItemUseRequest : public ParsedPacket {
public:
  ClientAgentInventoryItemUseRequest(const PacketContainer &packet);
  sro::scalar_types::StorageIndexType inventoryIndex() const;
  type_id::TypeId itemTypeId() const;
private:
  sro::scalar_types::StorageIndexType inventoryIndex_;
  type_id::TypeId itemTypeId_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_INVENTORY_ITEM_USE_REQUEST_HPP_