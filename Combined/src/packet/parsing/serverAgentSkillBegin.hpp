#ifndef PACKET_PARSING_SERVER_AGENT_SKILL_BEGIN_HPP
#define PACKET_PARSING_SERVER_AGENT_SKILL_BEGIN_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"
#include "../../pk2/skillData.hpp"

#include <cstdint>
// #include <string>

namespace packet::parsing {
  
class ServerAgentSkillBegin : public ParsedPacket {
public:
  ServerAgentSkillBegin(const PacketContainer &packet);
  uint8_t result() const;

  // 12292 Happens when trying to cast a skill and dont have enough MP
  // Not yet sure why 12293 happens. It's happening when we're knocked back and trying to cast an ice bolt
  //  Seems to happen when skill is already on cooldown
  // 12301 Happens when the wrong item is equipped
  uint16_t errorCode() const;

  uint32_t refSkillId() const;
  uint32_t casterGlobalId() const;
  uint32_t castId() const;
  uint32_t targetGlobalId() const;
  structures::SkillAction action() const;
private:
  uint8_t result_;
  uint16_t errorCode_;
  uint32_t refSkillId_;
  uint32_t casterGlobalId_;
  uint32_t castId_;
  uint32_t targetGlobalId_;
  structures::SkillAction action_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_SKILL_BEGIN_HPP