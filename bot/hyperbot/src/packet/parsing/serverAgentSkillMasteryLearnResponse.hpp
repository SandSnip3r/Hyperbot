#ifndef PACKET_PARSING_SERVER_AGENT_SKILL_MASTERY_LEARN_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_SKILL_MASTERY_LEARN_RESPONSE_HPP_

#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerAgentSkillMasteryLearnResponse : public ParsedPacket {
public:
  ServerAgentSkillMasteryLearnResponse(const PacketContainer &packet);
  bool success() const;
  uint32_t masteryId() const;
  uint8_t masteryLevel() const;
  uint16_t errorCode() const;
private:
  uint8_t result_;
  uint32_t masteryId_;
  uint8_t masteryLevel_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_SKILL_MASTERY_LEARN_RESPONSE_HPP_