#include "clientAgentInventoryItemUseRequest.hpp"

namespace packet::parsing {

ClientAgentInventoryItemUseRequest::ClientAgentInventoryItemUseRequest(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  inventoryIndex_ = stream.Read<sro::scalar_types::StorageIndexType>();
  itemTypeId_ = stream.Read<type_id::TypeId>();
}

sro::scalar_types::StorageIndexType ClientAgentInventoryItemUseRequest::inventoryIndex() const {
  return inventoryIndex_;
}

type_id::TypeId ClientAgentInventoryItemUseRequest::itemTypeId() const {
  return itemTypeId_;
}

} // namespace packet::parsing