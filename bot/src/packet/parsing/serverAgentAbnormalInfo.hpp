#ifndef PACKET_PARSING_SERVER_AGENT_ABNORMAL_INFO_HPP_
#define PACKET_PARSING_SERVER_AGENT_ABNORMAL_INFO_HPP_

#include "packet/parsing/parsedPacket.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include <array>
#include <cstdint>

namespace packet::parsing {

class ServerAgentAbnormalInfo : public ParsedPacket {
public:
  ServerAgentAbnormalInfo(const PacketContainer &packet);
  uint32_t stateBitmask() const { return stateBitmask_; }
  const std::array<packet::structures::vitals::AbnormalState, 32>& states() const { return states_; }
private:
  uint32_t stateBitmask_;
  std::array<packet::structures::vitals::AbnormalState, 32> states_ = {0};
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ABNORMAL_INFO_HPP_