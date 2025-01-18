#include "serverAgentResurrectOption.hpp"

namespace packet::parsing {

ServerAgentResurrectOption::ServerAgentResurrectOption(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(option_);
}

packet::enums::ResurrectionOptionFlag ServerAgentResurrectOption::option() const {
  return option_;
}

} // namespace packet::parsing