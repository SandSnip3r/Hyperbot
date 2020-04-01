#include "gameData.hpp"

#include "../../../common/pk2/pk2.h"
#include "../../../common/pk2/parsing/parsing.hpp"

#include <iostream>

namespace pk2 {

namespace fs = std::experimental::filesystem::v1;

GameData::GameData(const fs::path &kSilkroadPath) : kSilkroadPath_(kSilkroadPath) {
  try {
    auto kMediaPath = kSilkroadPath_ / "Media.pk2";
    Pk2ReaderModern pk2Reader{kMediaPath};
    parseMedia(pk2Reader);
  } catch (std::exception &ex) {
    throw std::runtime_error(std::string("Failed to parse Media.Pk2 at path \""+kSilkroadPath_.string()+"\". Error: \"")+ex.what()+"\"");
  }
}

void GameData::parseMedia(Pk2ReaderModern &pk2Reader) {
  parseDivisionInfo(pk2Reader);
  parseCharacterData(pk2Reader);
  parseItemData(pk2Reader);
  parseSkillData(pk2Reader);
  parseTeleportData(pk2Reader);
}

const DivisionInfo& GameData::divisionInfo() const {
  return divisionInfo_;
}

const CharacterData& GameData::characterData() const {
  return characterData_;
}

const ItemData& GameData::itemData() const {
  return itemData_;
}

const SkillData& GameData::skillData() const {
  return skillData_;
}

const TeleportData& GameData::teleportData() const {
  return teleportData_;
}

void GameData::parseDivisionInfo(Pk2ReaderModern &pk2Reader) {
  const std::string kDivisionInfoEntryName = "DIVISIONINFO.TXT";
  PK2Entry divisionInfoEntry = pk2Reader.getEntry(kDivisionInfoEntryName);
  auto divisionInfoData = pk2Reader.getEntryData(divisionInfoEntry);
  divisionInfo_ = parsing::parseDivisionInfo(divisionInfoData);
}

void GameData::parseCharacterData(Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterCharacterdataName = "characterdata.txt";
  const std::string kMasterCharacterdataPath = kTextdataDirectory + kMasterCharacterdataName;
  PK2Entry masterCharacterdataEntry = pk2Reader.getEntry(kMasterCharacterdataPath);

  auto masterCharacterdataData = pk2Reader.getEntryData(masterCharacterdataEntry);
  auto masterCharacterdataStr = parsing::fileDataToString(masterCharacterdataData);
  auto characterdataFilenames = parsing::split(masterCharacterdataStr, "\r\n");

  for (auto characterdataFilename : characterdataFilenames) {
    std::cout << "Parsing character data file \"" << characterdataFilename << "\"\n";
    auto characterdataPath = kTextdataDirectory + characterdataFilename;
    PK2Entry characterdataEntry = pk2Reader.getEntry(characterdataPath);
    auto characterdataData = pk2Reader.getEntryData(characterdataEntry);
    auto characterdataStr = parsing::fileDataToString(characterdataData);
    auto characterdataLines = parsing::split(characterdataStr, "\r\n");
    for (const auto &line : characterdataLines) {
      try {
        characterData_.addCharacter(parsing::parseCharacterdataLine(line));
      } catch (...) {
        std::cerr << "Failed to parse character data \"" << line << "\"\n";
      }
    }
  }
  std::cout << "Cached " << characterData_.size() << " character(s)\n";
}

void GameData::parseItemData(Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterItemdataName = "itemdata.txt";
  const std::string kMasterItemdataPath = kTextdataDirectory + kMasterItemdataName;
  PK2Entry masterItemdataEntry = pk2Reader.getEntry(kMasterItemdataPath);

  auto masterItemdataData = pk2Reader.getEntryData(masterItemdataEntry);
  auto masterItemdataStr = parsing::fileDataToString(masterItemdataData);
  auto itemdataFilenames = parsing::split(masterItemdataStr, "\r\n");

  for (auto itemdataFilename : itemdataFilenames) {
    std::cout << "Parsing item data file \"" << itemdataFilename << "\"\n";
    auto itemdataPath = kTextdataDirectory + itemdataFilename;
    PK2Entry itemdataEntry = pk2Reader.getEntry(itemdataPath);
    auto itemdataData = pk2Reader.getEntryData(itemdataEntry);
    auto itemdataStr = parsing::fileDataToString(itemdataData);
    auto itemdataLines = parsing::split(itemdataStr, "\r\n");
    for (const auto &line : itemdataLines) {
      try {
        itemData_.addItem(parsing::parseItemdataLine(line));
      } catch (...) {
        std::cerr << "Failed to parse item data \"" << line << "\"\n";
      }
    }
  }
  std::cout << "Cached " << itemData_.size() << " item(s)\n";
}

void GameData::parseSkillData(Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterSkilldataName = "skilldata.txt";
  const std::string kMasterSkilldataPath = kTextdataDirectory + kMasterSkilldataName;
  PK2Entry masterSkilldataEntry = pk2Reader.getEntry(kMasterSkilldataPath);

  auto masterSkilldataData = pk2Reader.getEntryData(masterSkilldataEntry);
  auto masterSkilldataStr = parsing::fileDataToString(masterSkilldataData);
  auto skilldataFilenames = parsing::split(masterSkilldataStr, "\r\n");

  for (auto skilldataFilename : skilldataFilenames) {
    std::cout << "Parsing skill data file \"" << skilldataFilename << "\"\n";
    auto skilldataPath = kTextdataDirectory + skilldataFilename;
    PK2Entry skilldataEntry = pk2Reader.getEntry(skilldataPath);
    auto skilldataData = pk2Reader.getEntryData(skilldataEntry);
    auto skilldataStr = parsing::fileDataToString(skilldataData);
    auto skilldataLines = parsing::split(skilldataStr, "\r\n");
    for (const auto &line : skilldataLines) {
      try {
        skillData_.addSkill(parsing::parseSkilldataLine(line));
      } catch (...) {
        std::cerr << "Failed to parse skill data \"" << line << "\"\n";
      }
    }
  }
  std::cout << "Cached " << skillData_.size() << " skill(s)\n";
}

void GameData::parseTeleportData(Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kTeleportDataFilename = "teleportbuilding.txt";
  std::cout << "Parsing teleport data file \"" << kTeleportDataFilename << "\"\n";
  auto teleportDataPath = kTextdataDirectory + kTeleportDataFilename;
  PK2Entry teleportDataEntry = pk2Reader.getEntry(teleportDataPath);
  auto teleportDataData = pk2Reader.getEntryData(teleportDataEntry);
  auto teleportDataStr = parsing::fileDataToString(teleportDataData);
  auto teleportDataLines = parsing::split(teleportDataStr, "\r\n");
  for (const auto &line : teleportDataLines) {
    try {
      teleportData_.addTeleport(parsing::parseTeleportbuildingLine(line));
    } catch (...) {
      std::cerr << "Failed to parse teleport data \"" << line << "\"\n";
    }
  }
  std::cout << "Cached " << teleportData_.size() << " teleport(s)\n";
}

} // namespace pk2