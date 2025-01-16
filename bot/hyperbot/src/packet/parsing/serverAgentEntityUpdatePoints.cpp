// #include "commonParsing.hpp"
#include "serverAgentEntityUpdatePoints.hpp"

namespace packet::parsing {

ServerAgentEntityUpdatePoints::ServerAgentEntityUpdatePoints(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  updatePointsType_ = static_cast<packet::enums::UpdatePointsType>(stream.Read<uint8_t>());
  if (updatePointsType_ == packet::enums::UpdatePointsType::kGold) {
    auto &goldUpdateData = updateData_.emplace<GoldUpdate>();
    stream.Read(goldUpdateData.gold);
    stream.Read<uint8_t>(goldUpdateData.isDisplayed);
  } else if (updatePointsType_ == packet::enums::UpdatePointsType::kSp) {
    auto &skillPointsUpdateData = updateData_.emplace<SkillPointsUpdate>();
    stream.Read(skillPointsUpdateData.skillPoints);
    stream.Read(skillPointsUpdateData.isDisplayed);
  } else if (updatePointsType_ == packet::enums::UpdatePointsType::kStatPoint) {
    auto &statPointsUpdateData = updateData_.emplace<StatPointsUpdate>();
    stream.Read(statPointsUpdateData.statPoints);
  } else if (updatePointsType_ == packet::enums::UpdatePointsType::kHwan) {
    auto &hwanPointsUpdateData = updateData_.emplace<HwanPointsUpdate>();
    stream.Read(hwanPointsUpdateData.hwanPoints);
    stream.Read(hwanPointsUpdateData.sourceGlobalId);
  } else if (updatePointsType_ == packet::enums::UpdatePointsType::kAp) {
    auto &apPointsUpdateData = updateData_.emplace<ApPointsUpdate>();
    stream.Read(apPointsUpdateData.apPoints);
  }
}

packet::enums::UpdatePointsType ServerAgentEntityUpdatePoints::updatePointsType() const {
  return updatePointsType_;
}

uint64_t ServerAgentEntityUpdatePoints::gold() const {
  return std::get<GoldUpdate>(updateData_).gold;
}

uint32_t ServerAgentEntityUpdatePoints::skillPoints() const {
  return std::get<SkillPointsUpdate>(updateData_).skillPoints;
}

bool ServerAgentEntityUpdatePoints::isDisplayed() const {
  if (updatePointsType_ == enums::UpdatePointsType::kGold) {
    return std::get<GoldUpdate>(updateData_).isDisplayed;
  } else {
    return std::get<SkillPointsUpdate>(updateData_).isDisplayed;
  }
}

uint16_t ServerAgentEntityUpdatePoints::statPoints() const {
  return std::get<StatPointsUpdate>(updateData_).statPoints;
}

uint8_t ServerAgentEntityUpdatePoints::hwanPoints() const {
  return std::get<HwanPointsUpdate>(updateData_).hwanPoints;
}

sro::scalar_types::EntityGlobalId ServerAgentEntityUpdatePoints::sourceGlobalId() const {
  return std::get<HwanPointsUpdate>(updateData_).sourceGlobalId;
}

uint32_t ServerAgentEntityUpdatePoints::apPoints() const {
  return std::get<ApPointsUpdate>(updateData_).apPoints;
}

} // namespace packet::parsing