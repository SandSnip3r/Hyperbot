#include "serverAgentEntityDamageEffect.hpp"

namespace packet::parsing {

ServerAgentEntityDamageEffect::ServerAgentEntityDamageEffect(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(globalId_);
  stream.Read(effectDamage_);
}

sro::scalar_types::EntityGlobalId ServerAgentEntityDamageEffect::globalId() const {
  return globalId_;
}

uint32_t ServerAgentEntityDamageEffect::effectDamage() const {
  return effectDamage_;
}

} // namespace packet::parsing