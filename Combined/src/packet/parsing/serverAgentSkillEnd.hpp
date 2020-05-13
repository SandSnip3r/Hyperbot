#ifndef PACKET_PARSING_SERVER_AGENT_SKILL_END_HPP
#define PACKET_PARSING_SERVER_AGENT_SKILL_END_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"
#include "../../pk2/skillData.hpp"

#include <cstdint>
// #include <string>

namespace packet::parsing {
  
class ServerAgentSkillEnd : public ParsedPacket {
public:
  ServerAgentSkillEnd(const PacketContainer &packet);
  uint8_t result() const;
  uint32_t castId() const;
  uint32_t targetGId() const;
  structures::SkillAction action() const;
  uint16_t errorCode() const;
private:
  uint8_t result_;
  uint32_t castId_;
  uint32_t targetGId_;
  structures::SkillAction action_;
  uint16_t errorCode_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_SKILL_END_HPP