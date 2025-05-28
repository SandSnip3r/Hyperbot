#include "serverAgentEntityUpdateStatus.hpp"
#include "helpers.hpp"

namespace packet::parsing {

ServerAgentEntityUpdateStatus::ServerAgentEntityUpdateStatus(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(entityUniqueId_);
  stream.Read(updateFlag_);
  stream.Read(vitalBitmask_);

  if (flags::isSet(vitalBitmask_, enums::VitalInfoFlag::kVitalInfoHp)) {
    stream.Read(newHpValue_);
  }

  if (flags::isSet(vitalBitmask_, enums::VitalInfoFlag::kVitalInfoMp)) {
    stream.Read(newMpValue_);
  }

  if (flags::isSet(vitalBitmask_, enums::VitalInfoFlag::kVitalInfoHgp)) {
    stream.Read(newHgpValue_);
  }

  if (flags::isSet(vitalBitmask_, enums::VitalInfoFlag::kVitalInfoAbnormal)) {
    stream.Read(stateBitmask_);
    for (uint32_t bitNum=helpers::toBitNum<enums::AbnormalStateFlag::kZombie>()+1; bitNum<32; ++bitNum) {
      const uint32_t flag = (uint32_t(1) << bitNum);
      if (stateBitmask_ & flag) {
        stream.Read(modernStateLevels_.at(bitNum));
      }
    }
  }
}

sro::scalar_types::EntityGlobalId ServerAgentEntityUpdateStatus::globalId() const {
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

const ServerAgentEntityUpdateStatus::ModernStateLevelArrayType& ServerAgentEntityUpdateStatus::modernStateLevels() const {
  return modernStateLevels_;
}

} // namespace packet::parsing