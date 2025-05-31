#include "serverGatewayLoginResponse.hpp"

#include "packet/enums/packetEnums.hpp"

namespace packet::building {

  PacketContainer ServerGatewayLoginResponse::success(uint32_t token, std::string_view agentServerIp, uint16_t agentServerPort) {
  StreamUtility stream;
  stream.Write(enums::LoginResult::kSuccess);
  stream.Write(token);
  stream.Write(agentServerIp);
  stream.Write(agentServerPort);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building