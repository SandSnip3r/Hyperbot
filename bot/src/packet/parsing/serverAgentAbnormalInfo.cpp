#include "helpers.hpp"
#include "serverAgentAbnormalInfo.hpp"

namespace packet::parsing {

ServerAgentAbnormalInfo::ServerAgentAbnormalInfo(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(stateBitmask_);
  for (int bitNum=0; bitNum<32; ++bitNum) {
    const uint32_t flag = (uint32_t(1) << bitNum);
    if (stateBitmask_ & flag) {
      packet::structures::vitals::AbnormalState &state = states_[bitNum];
      stream.Read(state.totalTime);
      stream.Read(state.timeElapsed);
      if (bitNum <= helpers::toBitNum<packet::enums::AbnormalStateFlag::kZombie>()) {
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