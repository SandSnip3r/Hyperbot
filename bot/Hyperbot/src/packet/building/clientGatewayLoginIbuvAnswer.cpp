#include "clientGatewayLoginIbuvAnswer.hpp"

namespace packet::building {

PacketContainer ClientGatewayLoginIbuvAnswer::packet(const std::string &answer) {
  StreamUtility stream;
  stream.Write<uint16_t>(answer.size());
  stream.Write_Ascii(answer);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building