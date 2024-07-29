#include "frameworkMessageIdentify.hpp"

namespace packet::parsing {

FrameworkMessageIdentify::FrameworkMessageIdentify(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(moduleName_);
  stream.Read(isCertified_);
}

} // namespace packet::parsing