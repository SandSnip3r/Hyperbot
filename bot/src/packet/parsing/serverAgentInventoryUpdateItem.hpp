#ifndef PACKET_PARSING_SERVER_AGENT_INVENTORY_UPDATE_ITEM_HPP_
#define PACKET_PARSING_SERVER_AGENT_INVENTORY_UPDATE_ITEM_HPP_

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"
#include "shared/silkroad_security.h"

namespace packet::parsing {

class ServerAgentInventoryUpdateItem : public ParsedPacket {
public:
  ServerAgentInventoryUpdateItem(const PacketContainer &packet);
  uint8_t slotIndex() const;
  bool itemUpdateHasFlag(enums::ItemUpdateFlag flag) const;
  uint16_t quantity() const;
private:
  uint8_t slotIndex_;
  enums::ItemUpdateFlag itemUpdateFlag_;
  uint16_t quantity_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_INVENTORY_UPDATE_ITEM_HPP_