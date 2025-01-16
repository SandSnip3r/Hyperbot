#include "clientAgentAuthRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentAuthRequest::packet(uint32_t loginToken, const std::string &kUsername, const std::string &kPassword, uint8_t kLocale, const std::array<uint8_t,6> &macAddress) {
  StreamUtility stream;
  stream.Write<uint32_t>(loginToken);
  stream.Write<uint16_t>(kUsername.size());
  stream.Write_Ascii(kUsername);
  stream.Write<uint16_t>(kPassword.size());
  stream.Write_Ascii(kPassword);
  stream.Write<uint8_t>(kLocale); //Content.ID
  for (const uint8_t macAddrByte : macAddress) {
    stream.Write<uint8_t>(macAddrByte);
  }
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building