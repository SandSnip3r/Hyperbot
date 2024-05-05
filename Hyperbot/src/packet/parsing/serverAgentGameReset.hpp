#ifndef PACKET_PARSING_SERVER_AGENT_GAME_RESET_HPP_
#define PACKET_PARSING_SERVER_AGENT_GAME_RESET_HPP_

#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerAgentGameReset : public ParsedPacket {
public:
  ServerAgentGameReset(const PacketContainer &packet);
private:
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_GAME_RESET_HPP_