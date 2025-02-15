#ifndef PACKET_PARSING_SERVER_AGENT_OPERATOR_RESPONSE_HPP_
#define PACKET_PARSING_SERVER_AGENT_OPERATOR_RESPONSE_HPP_

#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/parsedPacket.hpp"

namespace packet::parsing {

class ServerAgentOperatorResponse : public ParsedPacket {
public:
  ServerAgentOperatorResponse(const PacketContainer &packet);
  uint8_t result() const { return result_; }
  enums::OperatorCommand operatorCommand() const { return operatorCommand_; }
private:
  uint8_t result_;
  enums::OperatorCommand operatorCommand_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_OPERATOR_RESPONSE_HPP_