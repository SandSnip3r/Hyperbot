#ifndef PACKET_BUILDING_SERVER_GATEWAY_LOGIN_RESPONSE_HPP_
#define PACKET_BUILDING_SERVER_GATEWAY_LOGIN_RESPONSE_HPP_

#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#include <cstdint>
#include <string_view>

namespace packet::building {

class ServerGatewayLoginResponse {
private:
  static const Opcode kOpcode_ = Opcode::kServerGatewayLoginResponse;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer success(uint32_t token, std::string_view agentServerIp, uint16_t agentServerPort);

};

} // namespace packet::building

#endif // PACKET_BUILDING_SERVER_GATEWAY_LOGIN_RESPONSE_HPP_