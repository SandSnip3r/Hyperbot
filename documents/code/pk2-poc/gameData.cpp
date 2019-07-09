#include "gameData.hpp"

GameData::GameData(const std::string &clientDirectory) {
  // Parse Media.pk2
  parseMedia(clientDirectory);
}

void GameData::parseMedia(const std::string &clientDirectory) {
  const std::string kMediaPk2Path = clientDirectory + "/Media.pk2";
  bool openResult = pk2Reader_.Open(kMediaPk2Path);
  if (!openResult) {
    throw Pk2ReaderError("Unable to open pk2 file \""+kMediaPk2Path+"\"");
  }

  PK2Entry pk2Entry = {0};
  const std::string kDivisionInfoEntryName = "DIVISIONINFO.txt";
  bool getEntryResult = pk2Reader_.GetEntry(kDivisionInfoEntryName, pk2Entry);
  if (!getEntryResult) {
    throw Pk2ReaderError("Unable to get pk2 entry \""+kDivisionInfoEntryName+"\"");
  }
  // Use pk2Entry to extract data
  
  pk2Entry = PK2Entry{0};
  const std::string kItemDataEntryName = "itemData.txt";
  getEntryResult = pk2Reader_.GetEntry(kItemDataEntryName, pk2Entry);
  if (!getEntryResult) {
    throw Pk2ReaderError("Unable to get pk2 entry \""+kItemDataEntryName+"\"");
  }
  // Use pk2Entry to extract data
  
  // ...
  // Fill itemData_ and skillData_
}

GameData::~GameData() {

}

std::string GameData::gatewayAddress() const {

}

const ItemData& GameData::itemData() const {
  return itemData_;
}

const SkillData& GameData::skillData() const {
  return skillData_;
}

void GameData::initializeItemData() {

}

void GameData::initializeSkillData() {

}