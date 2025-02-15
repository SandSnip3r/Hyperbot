#include "serverAgentOperatorResponse.hpp"

namespace packet::parsing {

ServerAgentOperatorResponse::ServerAgentOperatorResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == 1) {
    // Success
    stream.Read(operatorCommand_);
    // TODO: There is more, but we don't care yet.
  } else if (result_ == 2) {
    // Error
    stream.Read(operatorCommand_);
    // TODO: There is more, but we don't care yet.
  }
}

} // namespace packet::parsing