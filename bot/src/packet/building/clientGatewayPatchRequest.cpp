#include "clientGatewayPatchRequest.hpp"

namespace packet::building {

PacketContainer ClientGatewayPatchRequest::packet(uint8_t locale, const std::string &moduleName, uint32_t version) {
  StreamUtility stream;
  stream.Write(locale);
  stream.Write(moduleName);
  stream.Write(version);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building