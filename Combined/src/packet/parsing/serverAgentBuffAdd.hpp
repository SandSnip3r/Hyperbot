#ifndef PACKET_PARSING_CLIENT_AGENT_BUFF_ADD_HPP
#define PACKET_PARSING_CLIENT_AGENT_BUFF_ADD_HPP

#include "parsedPacket.hpp"
#include "../../pk2/skillData.hpp"

#include <cstdint>
// #include <string>

namespace packet::parsing {
  
class ServerAgentBuffAdd : public ParsedPacket {
public:
  ServerAgentBuffAdd(const PacketContainer &packet, const pk2::SkillData &skillData);
  uint32_t globalId() const;
  uint32_t skillRefId() const;
  uint32_t activeBuffToken() const;
private:
  uint32_t globalId_;
  uint32_t skillRefId_;
  uint32_t activeBuffToken_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_BUFF_ADD_HPP