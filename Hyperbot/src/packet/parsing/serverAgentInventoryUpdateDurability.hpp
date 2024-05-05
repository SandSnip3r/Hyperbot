#ifndef PACKET_PARSING_SERVER_AGENT_INVENTORY_UPDATE_DURABILITY_HPP_
#define PACKET_PARSING_SERVER_AGENT_INVENTORY_UPDATE_DURABILITY_HPP_

#include "parsedPacket.hpp"

#include "../../shared/silkroad_security.h"

namespace packet::parsing {

class ServerAgentInventoryUpdateDurability : public ParsedPacket {
public:
  ServerAgentInventoryUpdateDurability(const PacketContainer &packet);
  uint8_t slotIndex() const;
  uint32_t durability() const;
private:
  uint8_t slotIndex_;
  uint32_t durability_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_INVENTORY_UPDATE_DURABILITY_HPP_