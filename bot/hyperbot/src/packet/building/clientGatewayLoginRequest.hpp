#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_GATEWAY_LOGIN_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_GATEWAY_LOGIN_REQUEST_HPP

namespace packet::building {

class ClientGatewayLoginRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientGatewayLoginRequest;
  static const bool kEncrypted_ = true;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint8_t locale, const std::string &username, const std::string &password, uint16_t shardId);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_GATEWAY_LOGIN_REQUEST_HPP