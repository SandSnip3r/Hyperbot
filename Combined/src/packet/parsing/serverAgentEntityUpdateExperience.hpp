#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_EXPERIENCE_HPP_
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_EXPERIENCE_HPP_

#include "parsedPacket.hpp"

#include <cstdint>

namespace packet::parsing {
  
class ServerAgentEntityUpdateExperience : public ParsedPacket {
public:
  ServerAgentEntityUpdateExperience(const PacketContainer &packet);
  uint64_t gainedExperiencePoints() const;
  uint64_t gainedSpExperiencePoints() const;
private:
  uint64_t gainedExperiencePoints_;
  uint64_t gainedSpExperiencePoints_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_EXPERIENCE_HPP_