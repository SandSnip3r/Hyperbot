#include "gameData.hpp"

#include "../../common/PK2.h"
#include "../../common/parsing.hpp"

namespace pk2::media {

namespace fs = std::experimental::filesystem::v1;

GameData::GameData(const fs::path &kSilkroadPath) : kSilkroadPath_(kSilkroadPath) {
  try {
    auto kMediaPath = kSilkroadPath_ / "Media.pk2";
    pk2::Pk2ReaderModern pk2Reader{kMediaPath};
    parseMedia(pk2Reader);
  } catch (std::exception &ex) {
    throw std::runtime_error(std::string("Failed to parse Media.Pk2. Error: \"")+ex.what()+"\"");
  }
}

void GameData::parseMedia(pk2::Pk2ReaderModern &pk2Reader) {
  parseDivisionInfo(pk2Reader);
  parseItemData(pk2Reader);
  parseSkillData(pk2Reader);
}

const pk2::DivisionInfo& GameData::divisionInfo() const {
  return divisionInfo_;
}

const ItemData& GameData::itemData() const {
  return itemData_;
}

const SkillData& GameData::skillData() const {
  return skillData_;
}

void GameData::parseDivisionInfo(pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kDivisionInfoEntryName = "DIVISIONINFO.TXT";
  PK2Entry divisionInfoEntry = pk2Reader.getEntry(kDivisionInfoEntryName);
  auto divisionInfoData = pk2Reader.getEntryData(divisionInfoEntry);
  divisionInfo_ = pk2::parsing::parseDivisionInfo(divisionInfoData);
}

void GameData::parseItemData(pk2::Pk2ReaderModern &pk2Reader) {

}

void GameData::parseSkillData(pk2::Pk2ReaderModern &pk2Reader) {

}

} // namespace pk2::media