#ifndef PACKET_PARSING_SERVER_GATEWAY_LOGIN_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_GATEWAY_LOGIN_RESPONSE_HPP_

#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerGatewayLoginResponse : public ParsedPacket {
public:
  ServerGatewayLoginResponse(const PacketContainer &packet);
  enums::LoginResult result() const { return result_; }
  uint32_t agentServerToken() const { return agentServerToken_; }
  std::string_view agentServerIp() const { return agentServerIp_; }
  uint16_t agentServerPort() const { return agentServerPort_; }
  enums::LoginErrorCode errorCode() const { return errorCode_; }
  uint32_t maxAttempts() const { return maxAttempts_; }
  uint32_t currentAttempts() const { return currentAttempts_; }
  enums::LoginBlockType blockType() const { return blockType_; }
  std::string_view punishmentReason() const { return punishmentReason_; }
  uint16_t punishmentEndDateYear() const { return punishmentEndDateYear_; }
  uint16_t punishmentEndDateMonth() const { return punishmentEndDateMonth_; }
  uint16_t punishmentEndDateDay() const { return punishmentEndDateDay_; }
  uint16_t punishmentEndDateHour() const { return punishmentEndDateHour_; }
  uint16_t punishmentEndDateMinute() const { return punishmentEndDateMinute_; }
  uint16_t punishmentEndDateSecond() const { return punishmentEndDateSecond_; }
  uint16_t punishmentEndDateMicrosecond() const { return punishmentEndDateMicrosecond_; }
private:
  enums::LoginResult result_;
  // Success case
  uint32_t agentServerToken_;
  std::string agentServerIp_;
  uint16_t agentServerPort_;
  
  // Error case
  enums::LoginErrorCode errorCode_;
  // Invalid login info error case
  uint32_t maxAttempts_;
  uint32_t currentAttempts_;
  // Blocked error case
  enums::LoginBlockType blockType_;
  // Punish blocked case
  std::string punishmentReason_;
  uint16_t punishmentEndDateYear_;
  uint16_t punishmentEndDateMonth_;
  uint16_t punishmentEndDateDay_;
  uint16_t punishmentEndDateHour_;
  uint16_t punishmentEndDateMinute_;
  uint16_t punishmentEndDateSecond_;
  uint16_t punishmentEndDateMicrosecond_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_GATEWAY_LOGIN_RESPONSE_HPP_