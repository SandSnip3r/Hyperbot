#include "clientGatewayLoginRequest.hpp"

namespace packet::building {

PacketContainer ClientGatewayLoginRequest::packet(uint8_t locale, const std::string &username, const std::string &password, uint16_t shardId) {
  StreamUtility stream;
  stream.Write<uint8_t>(locale);
  stream.Write<uint16_t>(username.size());
  stream.Write_Ascii(username);
  stream.Write<uint16_t>(password.size());
  stream.Write_Ascii(password);
  stream.Write<uint16_t>(shardId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building