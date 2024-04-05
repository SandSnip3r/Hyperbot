#ifndef PACKET_PARSING_SERVER_AGENT_BUFF_LINK_HPP_
#define PACKET_PARSING_SERVER_AGENT_BUFF_LINK_HPP_

#include "packet/parsing/parsedPacket.hpp"

#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <string>

namespace packet::parsing {

class ServerAgentBuffLink : public ParsedPacket {
public:
  ServerAgentBuffLink(const PacketContainer &packet);
  sro::scalar_types::ReferenceObjectId skillRefId() const;
  uint32_t activeBuffToken() const;
  sro::scalar_types::EntityGlobalId targetGlobalId() const;
  std::string targetName() const;
private:
  sro::scalar_types::ReferenceObjectId skillRefId_;
  uint32_t activeBuffToken_;
  sro::scalar_types::EntityGlobalId targetGlobalId_;
  std::string targetName_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_BUFF_LINK_HPP_