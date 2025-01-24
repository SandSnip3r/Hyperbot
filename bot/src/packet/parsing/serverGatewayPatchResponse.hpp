#ifndef PACKET_PARSING_SERVER_GATEWAY_PATCH_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_GATEWAY_PATCH_RESPONSE_HPP_

#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerGatewayPatchResponse : public ParsedPacket {
public:
  ServerGatewayPatchResponse(const PacketContainer &packet);
  uint8_t result() const { return result_; };
private:
  uint8_t result_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_GATEWAY_PATCH_RESPONSE_HPP_