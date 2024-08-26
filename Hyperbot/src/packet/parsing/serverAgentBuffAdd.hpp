#ifndef PACKET_PARSING_CLIENT_AGENT_BUFF_ADD_HPP
#define PACKET_PARSING_CLIENT_AGENT_BUFF_ADD_HPP

#include "parsedPacket.hpp"
#include "../../pk2/skillData.hpp"

#include <silkroad_lib/scalar_types.h>

#include <cstdint>

namespace packet::parsing {
  
class ServerAgentBuffAdd : public ParsedPacket {
public:
  ServerAgentBuffAdd(const PacketContainer &packet, const pk2::SkillData &skillData);
  sro::scalar_types::EntityGlobalId globalId() const { return globalId_; }
  sro::scalar_types::ReferenceObjectId skillRefId() const { return skillRefId_; }
  sro::scalar_types::BuffTokenType activeBuffToken() const { return activeBuffToken_; }
private:
  sro::scalar_types::EntityGlobalId globalId_;
  sro::scalar_types::ReferenceObjectId skillRefId_;
  sro::scalar_types::BuffTokenType activeBuffToken_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_CLIENT_AGENT_BUFF_ADD_HPP