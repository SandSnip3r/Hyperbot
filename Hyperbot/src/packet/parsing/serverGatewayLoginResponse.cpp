#include "serverGatewayLoginResponse.hpp"

namespace packet::parsing {

ServerGatewayLoginResponse::ServerGatewayLoginResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == enums::LoginResult::kSuccess) {
    stream.Read(agentServerToken_);
    stream.Read(agentServerIp_);
    stream.Read(agentServerPort_);
  } else if (result_ == enums::LoginResult::kFailed) {
    stream.Read(errorCode_);
    if (errorCode_ == enums::LoginErrorCode::kIncorrectUserInfo) {
      stream.Read(maxAttempts_);
      stream.Read(currentAttempts_);
    } else if (errorCode_ == enums::LoginErrorCode::kBlocked) {
      stream.Read(blockType_);
      if (blockType_ == enums::LoginBlockType::kPunishment) {
        stream.Read(punishmentReason_);
        stream.Read(punishmentEndDateYear_);
        stream.Read(punishmentEndDateMonth_);
        stream.Read(punishmentEndDateDay_);
        stream.Read(punishmentEndDateHour_);
        stream.Read(punishmentEndDateMinute_);
        stream.Read(punishmentEndDateSecond_);
        stream.Read(punishmentEndDateMicrosecond_);
      }
    }
  } else if (result_ == enums::LoginResult::kOther) {
    /* uint8_t unkByte0 = */ stream.Read<uint8_t>();
    /* uint8_t unkByte1 = */ stream.Read<uint8_t>();
    std::string errorMessage;
    stream.Read(errorMessage);
    LOG(INFO) << absl::StreamFormat("Error logging in: \"%s\"", errorMessage);
    /* uint16_t unkUShort0 = */ stream.Read<uint16_t>();
  }
}

} // namespace packet::parsing