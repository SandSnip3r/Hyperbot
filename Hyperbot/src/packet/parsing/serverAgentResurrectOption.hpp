#ifndef PACKET_PARSING_SERVER_AGENT_RESURRECT_OPTION_HPP_
#define PACKET_PARSING_SERVER_AGENT_RESURRECT_OPTION_HPP_

#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/parsedPacket.hpp"

#include <cstdint>

namespace packet::parsing {

class ServerAgentResurrectOption : public ParsedPacket {
public:
  ServerAgentResurrectOption(const PacketContainer &packet);
  packet::enums::ResurrectionOptionFlag option() const;
private:
  packet::enums::ResurrectionOptionFlag option_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_RESURRECT_OPTION_HPP_