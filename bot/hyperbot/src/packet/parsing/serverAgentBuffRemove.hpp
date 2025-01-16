#ifndef PACKET_PARSING_CLIENT_AGENT_BUFF_REMOVE_HPP
#define PACKET_PARSING_CLIENT_AGENT_BUFF_REMOVE_HPP

#include "parsedPacket.hpp"

#include <cstdint>
#include <vector>

namespace packet::parsing {
  
class ServerAgentBuffRemove : public ParsedPacket {
public:
  ServerAgentBuffRemove(const PacketContainer &packet);
  const std::vector<uint32_t>& tokens() const;
private:
  std::vector<uint32_t> tokens_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_BUFF_REMOVE_HPP