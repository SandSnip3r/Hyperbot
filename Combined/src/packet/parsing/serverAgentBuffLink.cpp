#include "serverAgentBuffLink.hpp"

namespace packet::parsing {

ServerAgentBuffLink::ServerAgentBuffLink(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(skillRefId_);
  stream.Read(activeBuffToken_);
  stream.Read(targetGlobalId_);
  stream.Read(targetName_);
}

sro::scalar_types::ReferenceObjectId ServerAgentBuffLink::skillRefId() const {
  return skillRefId_;
}

uint32_t ServerAgentBuffLink::activeBuffToken() const {
  return activeBuffToken_;
}

sro::scalar_types::EntityGlobalId ServerAgentBuffLink::targetGlobalId() const {
  return targetGlobalId_;
}

std::string ServerAgentBuffLink::targetName() const {
  return targetName_;
}

} // namespace packet::parsing