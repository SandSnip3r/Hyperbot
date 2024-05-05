#include "serverAgentInventoryItemUseResponse.hpp"

namespace packet::parsing {

ServerAgentInventoryItemUseResponse::ServerAgentInventoryItemUseResponse(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;

  result_ = stream.Read<uint8_t>();
  if (result_ == 1) {
    // Success
    slotNum_ = stream.Read<sro::scalar_types::StorageIndexType>();
    remainingCount_ = stream.Read<uint16_t>();
    typeData_ = stream.Read<type_id::TypeId>();
  } else {
    errorCode_ = static_cast<packet::enums::InventoryErrorCode>(stream.Read<uint16_t>());
  }
}

uint8_t ServerAgentInventoryItemUseResponse::result() const {
  return result_;
}

sro::scalar_types::StorageIndexType ServerAgentInventoryItemUseResponse::slotNum() const {
  return slotNum_;
}

uint16_t ServerAgentInventoryItemUseResponse::remainingCount() const {
  return remainingCount_;
}

type_id::TypeId ServerAgentInventoryItemUseResponse::typeData() const {
  return typeData_;
}

packet::enums::InventoryErrorCode ServerAgentInventoryItemUseResponse::errorCode() const {
  return errorCode_;
}

} // namespace packet::parsing