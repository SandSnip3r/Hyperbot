#ifndef PACKET_PARSING_SERVER_AGENT_INVENTORY_REPAIR_REPSONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_INVENTORY_REPAIR_REPSONSE_HPP_

#include "parsedPacket.hpp"
#include "shared/silkroad_security.h"

#include <optional>

namespace packet::parsing {

class ServerAgentInventoryRepairResponse : public ParsedPacket {
public:
  ServerAgentInventoryRepairResponse(const PacketContainer &packet);
  bool successful() const;
  uint16_t errorCode() const;
private:
  std::optional<uint16_t> errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_INVENTORY_REPAIR_REPSONSE_HPP_