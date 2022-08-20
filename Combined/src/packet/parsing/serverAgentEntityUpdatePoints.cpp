// #include "commonParsing.hpp"
#include "serverAgentEntityUpdatePoints.hpp"

namespace packet::parsing {

ServerAgentEntityUpdatePoints::ServerAgentEntityUpdatePoints(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  updatePointsType_ = static_cast<packet::enums::UpdatePointsType>(stream.Read<uint8_t>());
  if (updatePointsType_ == packet::enums::UpdatePointsType::kGold) {
    gold_ = stream.Read<uint64_t>();
    isDisplayed_ = stream.Read<uint8_t>();
  } else if (updatePointsType_ == packet::enums::UpdatePointsType::kSp) {
    skillPoints_ = stream.Read<uint32_t>();
    isDisplayed_ = stream.Read<uint8_t>();
  } else if (updatePointsType_ == packet::enums::UpdatePointsType::kStatPoint) {
    uint16_t StatPoints = stream.Read<uint16_t>();
  } else if (updatePointsType_ == packet::enums::UpdatePointsType::kHwan) {
    uint8_t HwanCount = stream.Read<uint8_t>();
    uint32_t Source_UniqueID = stream.Read<uint32_t>();
  } else if (updatePointsType_ == packet::enums::UpdatePointsType::kAp) {
    uint32_t APPoint = stream.Read<uint32_t>();
  }
}

packet::enums::UpdatePointsType ServerAgentEntityUpdatePoints::updatePointsType() const {
  return updatePointsType_;
}

uint64_t ServerAgentEntityUpdatePoints::gold() const {
  if (updatePointsType_ != packet::enums::UpdatePointsType::kGold) {
    throw std::runtime_error("Trying to get gold, but update type is not kGold");
  }
  return gold_;
}

uint32_t ServerAgentEntityUpdatePoints::skillPoints() const {
  return skillPoints_;
}

bool ServerAgentEntityUpdatePoints::isDisplayed() const {
  if (updatePointsType_ != packet::enums::UpdatePointsType::kGold) {
    throw std::runtime_error("Trying to get isDisplayed, but update type is not kGold");
  }
  return isDisplayed_;
}

} // namespace packet::parsing