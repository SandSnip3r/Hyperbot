#ifndef PACKET_PARSING_SERVER_AGENT_INVENTORY_EQUIP_COUNTDOWN_START_HPP_
#define PACKET_PARSING_SERVER_AGENT_INVENTORY_EQUIP_COUNTDOWN_START_HPP_

#include "packet/parsing/parsedPacket.hpp"

#include <silkroad_lib/scalar_types.hpp>

namespace packet::parsing {

class ServerAgentInventoryEquipCountdownStart : public ParsedPacket {
public:
  ServerAgentInventoryEquipCountdownStart(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId globalId() const { return globalId_; }
private:
  sro::scalar_types::EntityGlobalId globalId_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_INVENTORY_EQUIP_COUNTDOWN_START_HPP_