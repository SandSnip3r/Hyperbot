#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#ifndef PACKET_BUILDING_CLIENT_GATEWAY_LOGIN_IBUV_ANSWER_HPP
#define PACKET_BUILDING_CLIENT_GATEWAY_LOGIN_IBUV_ANSWER_HPP

namespace packet::building {

class ClientGatewayLoginIbuvAnswer {
private:
  static const Opcode kOpcode_ = Opcode::kClientGatewayLoginIbuvAnswer;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(const std::string &answer);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_GATEWAY_LOGIN_IBUV_ANSWER_HPP