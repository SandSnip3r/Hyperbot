#ifndef PACKET_PARSING_SERVER_AGENT_SKILL_LEARN_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_SKILL_LEARN_RESPONSE_HPP_

#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerAgentSkillLearnResponse : public ParsedPacket {
public:
  ServerAgentSkillLearnResponse(const PacketContainer &packet);
  bool success() const;
  uint32_t skillId() const;
  uint16_t errorCode() const;
private:
  uint8_t result_;
  uint32_t skillId_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_SKILL_LEARN_RESPONSE_HPP_