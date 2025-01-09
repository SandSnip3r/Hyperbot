#ifndef PACKET_PARSING_SERVER_AGENT_INVENTORY_OPERATION_REPSONSE_HPP
#define PACKET_PARSING_SERVER_AGENT_INVENTORY_OPERATION_REPSONSE_HPP

#include "parsedPacket.hpp"

#include "pk2/itemData.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include "../../shared/silkroad_security.h"

namespace packet::parsing {

class ServerAgentInventoryOperationResponse : public ParsedPacket {
public:
  ServerAgentInventoryOperationResponse(const PacketContainer &packet, const pk2::ItemData &itemData);
  const std::vector<structures::ItemMovement>& itemMovements() const;
private:
  std::vector<structures::ItemMovement> itemMovements_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_INVENTORY_OPERATION_REPSONSE_HPP