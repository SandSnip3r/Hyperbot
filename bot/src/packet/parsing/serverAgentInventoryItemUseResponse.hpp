#ifndef PACKET_PARSING_SERVER_AGENT_INVENTORY_ITEM_USE_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_INVENTORY_ITEM_USE_RESPONSE_HPP_

#include "packet/parsing/parsedPacket.hpp"
#include "type_id/typeCategory.hpp"

#include <silkroad_lib/scalar_types.hpp>

namespace packet::parsing {

class ServerAgentInventoryItemUseResponse : public ParsedPacket {
public:
  ServerAgentInventoryItemUseResponse(const PacketContainer &packet);
  uint8_t result() const;
  sro::scalar_types::StorageIndexType slotNum() const;
  uint16_t remainingCount() const;
  type_id::TypeId typeData() const;
  packet::enums::InventoryErrorCode errorCode() const;
private:
  uint8_t result_;
  sro::scalar_types::StorageIndexType slotNum_;
  uint16_t remainingCount_;
  type_id::TypeId typeData_;
  packet::enums::InventoryErrorCode errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_INVENTORY_ITEM_USE_RESPONSE_HPP_