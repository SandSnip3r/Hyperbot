#include "serverAgentAbnormalInfo.hpp"

namespace packet::parsing {

ServerAgentAbnormalInfo::ServerAgentAbnormalInfo(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(stateBitmask_);
  for (uint32_t i=0; i<32; ++i) {
    const auto bit = (1 << i);
    if (stateBitmask_ & bit) {
      packet::structures::vitals::AbnormalState &state = states_[i];
      stream.Read(state.totalTime);
      stream.Read(state.timeElapsed);
      if (bit <= static_cast<uint32_t>(packet::enums::AbnormalStateFlag::kZombie)) {
        // Legacy states
        stream.Read(state.effectOrLevel);
      } else {
        // Modern states
        stream.Read(state.effectOrLevel);
      }
    }
  }
}

} // namespace packet::parsing