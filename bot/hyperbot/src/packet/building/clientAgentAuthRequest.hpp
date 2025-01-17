#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#include <array>
#include <string>

#ifndef PACKET_BUILDING_CLIENT_AGENT_AUTH_REQUEST_HPP
#define PACKET_BUILDING_CLIENT_AGENT_AUTH_REQUEST_HPP

namespace packet::building {

class ClientAgentAuthRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientAgentAuthRequest;
  static const bool kEncrypted_ = true;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint32_t loginToken, const std::string &kUsername, const std::string &kPassword, uint8_t kLocale, const std::array<uint8_t,6> &macAddress);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_AGENT_AUTH_REQUEST_HPP