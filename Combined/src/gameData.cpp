#include "gameData.hpp"

#include "../../common/PK2.h"
#include "../../common/parsing.hpp"

#include <iostream>

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
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterItemdataName = "itemdata.txt";
  const std::string kMasterItemdataPath = kTextdataDirectory + kMasterItemdataName;
  PK2Entry masterItemdataEntry = pk2Reader.getEntry(kMasterItemdataPath);

  auto masterItemdataData = pk2Reader.getEntryData(masterItemdataEntry);
  auto masterItemdataStr = pk2::parsing::fileDataToString(masterItemdataData);
  auto itemdataFilenames = pk2::parsing::split(masterItemdataStr, "\r\n");

  for (auto itemdataFilename : itemdataFilenames) {
    std::cout << "Parsing item data file \"" << itemdataFilename << "\"\n";
    auto itemdataPath = kTextdataDirectory + itemdataFilename;
    PK2Entry itemdataEntry = pk2Reader.getEntry(itemdataPath);
    auto itemdataData = pk2Reader.getEntryData(itemdataEntry);
    auto itemdataStr = pk2::parsing::fileDataToString(itemdataData);
    auto itemdataLines = pk2::parsing::split(itemdataStr, "\r\n");
    for (const auto &line : itemdataLines) {
      try {
        itemData_.addItem(pk2::parsing::parseItemdataLine(line));
      } catch (...) {
        std::cerr << "Failed to parse item data \"" << line << "\"\n";
      }
    }
  }
  std::cout << "Cached " << itemData_.size() << " item(s)\n";
}

void GameData::parseSkillData(pk2::Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterSkilldataName = "skilldata.txt";
  const std::string kMasterSkilldataPath = kTextdataDirectory + kMasterSkilldataName;
  PK2Entry masterSkilldataEntry = pk2Reader.getEntry(kMasterSkilldataPath);

  auto masterSkilldataData = pk2Reader.getEntryData(masterSkilldataEntry);
  auto masterSkilldataStr = pk2::parsing::fileDataToString(masterSkilldataData);
  auto skilldataFilenames = pk2::parsing::split(masterSkilldataStr, "\r\n");

  for (auto skilldataFilename : skilldataFilenames) {
    std::cout << "Parsing skill data file \"" << skilldataFilename << "\"\n";
    auto skilldataPath = kTextdataDirectory + skilldataFilename;
    PK2Entry skilldataEntry = pk2Reader.getEntry(skilldataPath);
    auto skilldataData = pk2Reader.getEntryData(skilldataEntry);
    auto skilldataStr = pk2::parsing::fileDataToString(skilldataData);
    auto skilldataLines = pk2::parsing::split(skilldataStr, "\r\n");
    for (const auto &line : skilldataLines) {
      try {
        skillData_.addSkill(pk2::parsing::parseSkilldataLine(line));
      } catch (...) {
        std::cerr << "Failed to parse skill data \"" << line << "\"\n";
      }
    }
  }
  std::cout << "Cached " << skillData_.size() << " skill(s)\n";
}

} // namespace pk2::media