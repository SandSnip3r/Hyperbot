#ifndef PACKET_PARSING_SERVER_AGENT_INVENTORY_OPERATION_REPSONSE_HPP
#define PACKET_PARSING_SERVER_AGENT_INVENTORY_OPERATION_REPSONSE_HPP

#include "parsedPacket.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "shared/silkroad_security.h"

#include <silkroad_lib/pk2/itemData.hpp>

namespace packet::parsing {

class ServerAgentInventoryOperationResponse : public ParsedPacket {
public:
  ServerAgentInventoryOperationResponse(const PacketContainer &packet, const sro::pk2::ItemData &itemData);
  const std::vector<structures::ItemMovement>& itemMovements() const;
  bool success() const { return result_ == 1; }
  uint16_t errorCode() const { return errorCode_; }
private:
  uint8_t result_;
  uint16_t errorCode_;
  std::vector<structures::ItemMovement> itemMovements_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_INVENTORY_OPERATION_REPSONSE_HPP
