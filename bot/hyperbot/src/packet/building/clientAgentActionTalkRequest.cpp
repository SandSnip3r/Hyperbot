#include "clientAgentActionTalkRequest.hpp"

#include <type_traits>

namespace packet::building {

PacketContainer ClientAgentActionTalkRequest::packet(uint32_t gId, packet::enums::TalkOption talkOption) {
  StreamUtility stream;
  stream.Write<>(gId);
  stream.Write<>(static_cast<std::underlying_type_t<packet::enums::TalkOption>>(talkOption));
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building