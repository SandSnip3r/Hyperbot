#include "serverAgentActionTalkResponse.hpp"

#include "logging.hpp"

namespace packet::parsing {

ServerAgentActionTalkResponse::ServerAgentActionTalkResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  if (result_ == 1) {
    talkOption_ = static_cast<packet::enums::TalkOption>(stream.Read<uint8_t>());
    if (talkOption_ == packet::enums::TalkOption::kTrade) {
      // Special trader shops with Bargain goods has started.
      isSpecialtyTime_ = stream.Read<uint8_t>();
    }
  } else {
    errorCode_ = stream.Read<uint16_t>();
  }
}

uint8_t ServerAgentActionTalkResponse::result() const {
  return result_;
}

packet::enums::TalkOption ServerAgentActionTalkResponse::talkOption() const {
  return talkOption_;
}

uint8_t ServerAgentActionTalkResponse::isSpecialtyTime() const {
  return isSpecialtyTime_;
}

uint16_t ServerAgentActionTalkResponse::errorCode() const {
  return errorCode_;
}

} // namespace packet::parsing