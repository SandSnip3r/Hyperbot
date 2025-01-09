#include "serverAgentInventoryRepairResponse.hpp"

namespace packet::parsing {

ServerAgentInventoryRepairResponse::ServerAgentInventoryRepairResponse(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  const uint8_t result_ = stream.Read<uint8_t>();
  if (result_ != 1) {
    // Failed
    errorCode_ = stream.Read<uint16_t>();
  }
}

bool ServerAgentInventoryRepairResponse::successful() const {
  return !errorCode_.has_value();
}

uint16_t ServerAgentInventoryRepairResponse::errorCode() const {
  if (!errorCode_) {
    throw std::runtime_error("Trying to get error code, but repair was successful");
  }
  return *errorCode_;
}

} // namespace packet::parsing