#ifndef PACKET_PARSING_SERVER_AGENT_CHARACTER_UPDATE_STATS_HPP_
#define PACKET_PARSING_SERVER_AGENT_CHARACTER_UPDATE_STATS_HPP_

#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerAgentCharacterUpdateStats : public ParsedPacket {
public:
  ServerAgentCharacterUpdateStats(const PacketContainer &packet);
  uint32_t maxHp() const { return maxHp_; }
  uint32_t maxMp() const { return maxMp_; }
  uint16_t strPoints() const { return strPoints_; }
  uint16_t intPoints() const { return intPoints_; }
private:
  uint32_t maxHp_;
  uint32_t maxMp_;
  uint16_t strPoints_;
  uint16_t intPoints_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_CHARACTER_UPDATE_STATS_HPP_