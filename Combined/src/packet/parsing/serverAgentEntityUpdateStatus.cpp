#include "serverAgentEntityUpdateStatus.hpp"

namespace packet::parsing {

ServerAgentEntityUpdateStatus::ServerAgentEntityUpdateStatus(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;

  entityUniqueId_ = stream.Read<sro::scalar_types::EntityGlobalId>();
  updateFlag_ = static_cast<enums::UpdateFlag>(stream.Read<uint16_t>());
  vitalBitmask_ = stream.Read<enums::VitalInfoFlag>();

  if (flags::isSet(vitalBitmask_, enums::VitalInfoFlag::kVitalInfoHp)) {
    newHpValue_ = stream.Read<uint32_t>();
  }

  if (flags::isSet(vitalBitmask_, enums::VitalInfoFlag::kVitalInfoMp)) {
    newMpValue_ = stream.Read<uint32_t>();
  }

  if (flags::isSet(vitalBitmask_, enums::VitalInfoFlag::kVitalInfoHgp)) {
    newHgpValue_ = stream.Read<uint16_t>();
  }

  if (flags::isSet(vitalBitmask_, enums::VitalInfoFlag::kVitalInfoAbnormal)) {
    stateBitmask_ = stream.Read<uint32_t>();
    for (uint32_t i=0; i<32; ++i) {
      const auto bit = (1 << i);
      if (bit > static_cast<uint32_t>(enums::AbnormalStateFlag::kZombie) && stateBitmask_ & bit) {
        stateLevels_.push_back(stream.Read<uint8_t>());
      }
    }
  }
}

sro::scalar_types::EntityGlobalId ServerAgentEntityUpdateStatus::entityUniqueId() const {
  return entityUniqueId_;
}

enums::UpdateFlag ServerAgentEntityUpdateStatus::updateFlag() const {
  return updateFlag_;
}

enums::VitalInfoFlag ServerAgentEntityUpdateStatus::vitalBitmask() const {
  return vitalBitmask_;
}

uint32_t ServerAgentEntityUpdateStatus::newHpValue() const {
  return newHpValue_;
}

uint32_t ServerAgentEntityUpdateStatus::newMpValue() const {
  return newMpValue_;
}

uint16_t ServerAgentEntityUpdateStatus::newHgpValue() const {
  return newHgpValue_;
}

uint32_t ServerAgentEntityUpdateStatus::stateBitmask() const {
  return stateBitmask_;
}

const std::vector<uint8_t>& ServerAgentEntityUpdateStatus::stateLevels() const {
  return stateLevels_;
}

} // namespace packet::parsing