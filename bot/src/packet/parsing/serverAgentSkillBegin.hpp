#ifndef PACKET_PARSING_SERVER_AGENT_SKILL_BEGIN_HPP
#define PACKET_PARSING_SERVER_AGENT_SKILL_BEGIN_HPP

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"
#include "pk2/skillData.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>

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

  sro::scalar_types::ReferenceObjectId refSkillId() const;
  sro::scalar_types::EntityGlobalId casterGlobalId() const;
  uint32_t castId() const;
  sro::scalar_types::EntityGlobalId targetGlobalId() const;
  structures::SkillAction action() const;
private:
  uint8_t result_;
  uint16_t errorCode_;
  sro::scalar_types::ReferenceObjectId refSkillId_;
  sro::scalar_types::EntityGlobalId casterGlobalId_;
  uint32_t castId_;
  sro::scalar_types::EntityGlobalId targetGlobalId_;
  structures::SkillAction action_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_SKILL_BEGIN_HPP