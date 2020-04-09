#include "serverAgentBuffRemove.hpp"

namespace packet::parsing {

ServerAgentBuffRemove::ServerAgentBuffRemove(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint8_t tokenCount = stream.Read<uint8_t>();
  for (uint8_t tokenNum=0; tokenNum<tokenCount; ++tokenNum) {
    tokens_.emplace_back(stream.Read<uint32_t>());
  }
}

const std::vector<uint32_t>& ServerAgentBuffRemove::tokens() const {
  return tokens_;
}

} // namespace packet::parsing