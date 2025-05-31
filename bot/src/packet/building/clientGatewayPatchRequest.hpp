#ifndef PACKET_BUILDING_CLIENT_GATEWAY_PATCH_REQUEST_HPP_
#define PACKET_BUILDING_CLIENT_GATEWAY_PATCH_REQUEST_HPP_

#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

#include <cstdint>
#include <string>

namespace packet::building {

class ClientGatewayPatchRequest {
private:
  static const Opcode kOpcode_ = Opcode::kClientGatewayPatchRequest;
  static const bool kEncrypted_ = true;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet(uint8_t locale, const std::string &moduleName, uint32_t version);
};

} // namespace packet::building

#endif // PACKET_BUILDING_CLIENT_GATEWAY_PATCH_REQUEST_HPP_