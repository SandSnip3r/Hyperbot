#ifndef PACKET_PARSING_SERVER_GATEWAY_LOGIN_IBUV_CHALLENGE_HPP_
#define PACKET_PARSING_SERVER_GATEWAY_LOGIN_IBUV_CHALLENGE_HPP_

#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerGatewayLoginIbuvChallenge : public ParsedPacket {
public:
  ServerGatewayLoginIbuvChallenge(const PacketContainer &packet);
private:
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_GATEWAY_LOGIN_IBUV_CHALLENGE_HPP_