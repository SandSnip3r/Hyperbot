#include "clientAgentActionTalkRequest.hpp"

namespace packet::parsing {

ClientAgentActionTalkRequest::ClientAgentActionTalkRequest(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  gId_ = stream.Read<uint32_t>();
  talkOption_ = static_cast<enums::TalkOption>(stream.Read<uint8_t>());
}

uint32_t ClientAgentActionTalkRequest::gId() const {
  return gId_;
}

enums::TalkOption ClientAgentActionTalkRequest::talkOption() const {
  return talkOption_;
}

} // namespace packet::parsing