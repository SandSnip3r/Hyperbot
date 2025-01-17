#include "gameData.hpp"

#include "math_helpers.h"

#include <silkroad_lib/pk2/navmeshParser.h>
#include <silkroad_lib/pk2/parsing/parsing.h>
#include <silkroad_lib/pk2/parsing/regionInfoParser.h>
#include <silkroad_lib/pk2/pk2.h>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>

#include <functional>
#include <fstream>
#include <map>
#include <string_view>
#include <thread>
#include <vector>

namespace pk2 {

void GameData::parseSilkroadFiles(const std::filesystem::path &clientPath) {
  LOG(INFO) << "Parsing Silkroad files";
  try {
    const auto kDataPath = clientPath / "Data.pk2";
    sro::pk2::Pk2ReaderModern pk2Reader{kDataPath};
    parseData(pk2Reader);
  } catch (std::exception &ex) {
    throw std::runtime_error(std::string("Failed to parse Data.Pk2 at path \""+clientPath.string()+"\". Error: \"")+ex.what()+"\"");
  }
  try {
    const auto kMediaPath = clientPath / "Media.pk2";
    sro::pk2::Pk2ReaderModern pk2Reader{kMediaPath};
    parseMedia(pk2Reader);
  } catch (std::exception &ex) {
    throw std::runtime_error(std::string("Failed to parse Media.Pk2 at path \""+clientPath.string()+"\". Error: \"")+ex.what()+"\"");
  }
  LOG(INFO) << "Done parsing Silkroad files";
}

void GameData::parseData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  VLOG(1) << "Parsing Data.pk2";
  // TODO: We're not currently using navmesh data. Once the memory leak in triangle.c is resolved, reenable this.
  // parseNavmeshData(pk2Reader);
  parseRegionInfo(pk2Reader);
  VLOG(1) << "Done parsing Data.pk2";
}

void GameData::parseMedia(sro::pk2::Pk2ReaderModern &pk2Reader) {
  VLOG(1) << "Parsing Media.pk2";
  std::vector<std::thread> thrs;
  VLOG(2) << "Parsing Gateway Port";
  parseGatewayPort(pk2Reader);
  VLOG(2) << "Parsing Division Info";
  parseDivisionInfo(pk2Reader);
  VLOG(2) << "Parsing Shop Data";
  parseShopData(pk2Reader);
  VLOG(2) << "Parsing Magic Option Data";
  parseMagicOptionData(pk2Reader);
  VLOG(2) << "Parsing Level Data";
  parseLevelData(pk2Reader);
  VLOG(2) << "Parsing Ref Region";
  parseRefRegion(pk2Reader);
  VLOG(2) << "Parsing Text Data";
  parseTextData(pk2Reader);
  VLOG(2) << "Parsing Character, Item, Skill, and Teleport Data in multiple threads";
  thrs.emplace_back(&GameData::parseCharacterData, this, std::ref(pk2Reader));
  thrs.emplace_back(&GameData::parseItemData, this, std::ref(pk2Reader));
  thrs.emplace_back(&GameData::parseSkillData, this, std::ref(pk2Reader));
  parseTeleportData(pk2Reader);
  for (auto &thr : thrs) {
    thr.join();
  }
  VLOG(1) << "Done parsing Media.pk2";
}

const uint16_t GameData::gatewayPort() const {
  return gatewayPort_;
}

const sro::pk2::DivisionInfo& GameData::divisionInfo() const {
  return divisionInfo_;
}

const CharacterData& GameData::characterData() const {
  return characterData_;
}

const ItemData& GameData::itemData() const {
  return itemData_;
}

const ShopData& GameData::shopData() const {
  return shopData_;
}

const SkillData& GameData::skillData() const {
  return skillData_;
}

const MasteryData& GameData::masteryData() const {
  return masteryData_;
}

const MagicOptionData& GameData::magicOptionData() const {
  return magicOptionData_;
}

const LevelData& GameData::levelData() const {
  return levelData_;
}

const RefRegion& GameData::refRegion() const {
  return refRegion_;
}

const TextData& GameData::textData() const {
  return textData_;
}

const TextZoneNameData& GameData::textZoneNameData() const {
  return textZoneNameData_;
}

const TeleportData& GameData::teleportData() const {
  return teleportData_;
}

const sro::navmesh::triangulation::NavmeshTriangulation& GameData::navmeshTriangulation() const {
  if (!navmeshTriangulation_.has_value()) {
    throw std::runtime_error("Asking for navmesh triangulation which does not exist");
  }
  return navmeshTriangulation_.value();
}

const sro::pk2::RegionInfo& GameData::regionInfo() const {
  return regionInfo_;
}

std::string GameData::getSkillName(sro::scalar_types::ReferenceObjectId skillRefId) const {
  const sro::pk2::ref::Skill &skill = skillData_.getSkillById(skillRefId);
  const std::optional<std::string> maybeName = textData_.getSkillNameIfExists(skill.uiSkillName);
  return (maybeName ? *maybeName : "UNKNOWN_SKILL");
}

std::string GameData::getItemName(sro::scalar_types::ReferenceObjectId itemRefId) const {
  const sro::pk2::ref::Item &item = itemData_.getItemById(itemRefId);
  const std::optional<std::string> maybeName = textData_.getItemNameIfExists(item.nameStrID128);
  return (maybeName ? *maybeName : "UNKNOWN_ITEM");
}

std::string GameData::getMasteryName(sro::pk2::ref::MasteryId masteryId) const {
  const sro::pk2::ref::Mastery &masteryDataObj = masteryData_.getMasteryById(masteryId);
  const std::optional<std::string> maybeName = textData_.getMasteryNameIfExists(masteryDataObj.masteryNameCode);
  return (maybeName ? *maybeName : "UNKNOWN_MASTERY");
}

sro::pk2::ref::MasteryId GameData::getMasteryId(std::string masteryName) const {
  std::transform(masteryName.begin(), masteryName.end(), masteryName.begin(), [](unsigned char c) { return std::tolower(c); });
  masteryName[0] = std::toupper(masteryName[0]);
  std::optional<std::string> maybeNameCode = textData_.getMasteryNameCodeIfExists(masteryName);
  if (!maybeNameCode) {
    throw std::runtime_error(absl::StrFormat("Cannot get mastery id for mastery \"%s\"", masteryName));
  }
  return masteryData_.getMasteryIdByMasteryNameCode(*maybeNameCode);
}

void GameData::parseGatewayPort(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kGatePortEntryName = "GATEPORT.TXT";
  sro::pk2::PK2Entry gatePortEntry = pk2Reader.getEntry(kGatePortEntryName);
  auto gatePortData = pk2Reader.getEntryData(gatePortEntry);
  gatewayPort_ = sro::pk2::parsing::parseGatePort(gatePortData);
}

void GameData::parseDivisionInfo(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kDivisionInfoEntryName = "DIVISIONINFO.TXT";
  sro::pk2::PK2Entry divisionInfoEntry = pk2Reader.getEntry(kDivisionInfoEntryName);
  auto divisionInfoData = pk2Reader.getEntryData(divisionInfoEntry);
  divisionInfo_ = sro::pk2::parsing::parseDivisionInfo(divisionInfoData);
}

namespace {
template<typename DataType>
void parseDataFile(const std::string &data,
                   std::function<bool(const std::string &)> isValidDataLine,
                   std::function<DataType(const std::string &)> parseDataLine,
                   std::function<void(DataType &&)> saveParsedDataObject) {
  size_t start=0;
  size_t posOfNewline = data.find('\n');
  while ((posOfNewline = data.find('\n', start)) != std::string::npos) {
    int carriageReturnOffset = 0;
    if (data[posOfNewline-1] == '\r') {
      // Dont add carriage return to the "line"
      carriageReturnOffset = 1;
    }
    const auto line = data.substr(start, posOfNewline-start-carriageReturnOffset);
    if (isValidDataLine(line)) {
      saveParsedDataObject(parseDataLine(line));
    }
    start = posOfNewline+1;
  }
  if (start < data.size()-1) {
    // File doesnt end in newline. One more line to read
    LOG(INFO) << "One more\n";
    const auto line = data.substr(start);
    if (isValidDataLine(line)) {
      saveParsedDataObject(parseDataLine(line));
    }
  }
}

template<typename DataType>
void parseDataFile2(const std::vector<std::string> &lines,
                   std::function<bool(const std::string &)> isValidDataLine,
                   std::function<DataType(const std::string &)> parseDataLine,
                   std::function<void(DataType &&)> saveParsedDataObject) {
  for (const auto &line : lines) {
    if (isValidDataLine(line)) {
      saveParsedDataObject(parseDataLine(line));
    }
  }
}

std::string getFileDataAsString(sro::pk2::Pk2ReaderModern &pk2Reader, const std::string &path) {
  sro::pk2::PK2Entry entry = pk2Reader.getEntry(path);
  auto entryData = pk2Reader.getEntryData(entry);
  return sro::pk2::parsing::fileDataToString(entryData);
}

} // namespace (anonymous)

void GameData::parseCharacterData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterCharacterdataName = "characterdata.txt";
  const std::string kMasterCharacterdataPath = kTextdataDirectory + kMasterCharacterdataName;
  sro::pk2::PK2Entry masterCharacterdataEntry = pk2Reader.getEntry(kMasterCharacterdataPath);

  auto masterCharacterdataData = pk2Reader.getEntryData(masterCharacterdataEntry);
  // auto masterCharacterdataStr = sro::pk2::parsing::fileDataToString(masterCharacterdataData);
  // auto characterdataFilenames = sro::pk2::parsing::split(masterCharacterdataStr, "\r\n");
  auto characterdataFilenames = sro::pk2::parsing::fileDataToStringLines(masterCharacterdataData);

  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing character data";
  }
  for (auto characterdataFilename : characterdataFilenames) {
    auto characterdataPath = kTextdataDirectory + characterdataFilename;
    sro::pk2::PK2Entry characterdataEntry = pk2Reader.getEntry(characterdataPath);
    auto characterdataData = pk2Reader.getEntryData(characterdataEntry);
    // auto characterdataStr = sro::pk2::parsing::fileDataToString(characterdataData);
    auto characterdataLines = sro::pk2::parsing::fileDataToStringLines(characterdataData);
    try {
      parseDataFile2<sro::pk2::ref::Character>(characterdataLines, sro::pk2::parsing::isValidCharacterdataLine, sro::pk2::parsing::parseCharacterdataLine, std::bind(&CharacterData::addCharacter, &characterData_, std::placeholders::_1));
      // parseDataFile<sro::pk2::ref::Character>(characterdataStr, sro::pk2::parsing::isValidCharacterdataLine, sro::pk2::parsing::parseCharacterdataLine, std::bind(&CharacterData::addCharacter, &characterData_, std::placeholders::_1));
    } catch (std::exception &ex) {
      LOG(WARNING) << "Exception while parsing character data: \"" << ex.what() << '"';
    }
  }
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << characterData_.size() << " character(s)";
  }
}

void GameData::parseItemData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterItemdataName = "itemdata.txt";
  const std::string kMasterItemdataPath = kTextdataDirectory + kMasterItemdataName;
  auto masterItemdataStr = getFileDataAsString(pk2Reader, kMasterItemdataPath);
  auto itemdataFilenames = sro::pk2::parsing::split(masterItemdataStr, "\r\n");

  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing item data";
  }
  for (auto itemdataFilename : itemdataFilenames) {
    auto itemdataPath = kTextdataDirectory + itemdataFilename;
    auto itemdataStr = getFileDataAsString(pk2Reader, itemdataPath);
    parseDataFile<sro::pk2::ref::Item>(itemdataStr, sro::pk2::parsing::isValidItemdataLine, sro::pk2::parsing::parseItemdataLine, std::bind(&ItemData::addItem, &itemData_, std::placeholders::_1));
  }
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << itemData_.size() << " item(s)";
  }
}

void GameData::parseSkillData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  // Prefer the encrypted file, as the client uses this and not the unencrypted version (skilldata.txt)
  const std::string kMasterSkilldataName = "skilldataenc.txt";
  const std::string kMasterSkilldataPath = kTextdataDirectory + kMasterSkilldataName;
  auto masterSkilldataStr = getFileDataAsString(pk2Reader, kMasterSkilldataPath);
  auto skilldataFilenames = sro::pk2::parsing::split(masterSkilldataStr, "\r\n");

  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing skill data";
  }
  for (auto skilldataFilename : skilldataFilenames) {
    auto skilldataPath = kTextdataDirectory + skilldataFilename;
    sro::pk2::PK2Entry skilldataEntry = pk2Reader.getEntry(skilldataPath);
    auto skilldataData = pk2Reader.getEntryData(skilldataEntry);
    // Decrypt this skill data
    sro::pk2::parsing::decryptSkillData(skilldataData);
    auto skilldataStr = sro::pk2::parsing::fileDataToString(skilldataData);
    parseDataFile<sro::pk2::ref::Skill>(skilldataStr, sro::pk2::parsing::isValidSkilldataLine, sro::pk2::parsing::parseSkilldataLine, std::bind(&SkillData::addSkill, &skillData_, std::placeholders::_1));
  }
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << skillData_.size() << " skill(s)";
  }
}

void GameData::parseTeleportData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kTeleportDataFilename = "teleportbuilding.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing teleport data.";
  }
  auto teleportDataPath = kTextdataDirectory + kTeleportDataFilename;
  auto teleportDataStr = getFileDataAsString(pk2Reader, teleportDataPath);
  parseDataFile<sro::pk2::ref::Teleport>(teleportDataStr, sro::pk2::parsing::isValidTeleportbuildingLine, sro::pk2::parsing::parseTeleportbuildingLine, std::bind(&TeleportData::addTeleport, &teleportData_, std::placeholders::_1));
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << teleportData_.size() << " teleport(s)";
  }
}

void GameData::parseShopData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  // Maps Package name to item data
  std::map<std::string, sro::pk2::ref::ScrapOfPackageItem> scrapOfPackageItemMap;
  // Maps a ShopTabGroup(I think an option within the NPC's dialog) to N tabs
  std::vector<sro::pk2::ref::ShopTab> shopTabs;
  // List of NPC->ShopGroup
  std::vector<sro::pk2::ref::ShopGroup> shopGroups;
  // List of Tab->Good
  std::vector<sro::pk2::ref::ShopGood> shopGoods;
  // List of ShopGroup->Shop
  std::vector<sro::pk2::ref::MappingShopGroup> mappingShopGroups;
  // List of Shop->ShopTabGroup
  std::vector<sro::pk2::ref::MappingShopWithTab> mappingShopWithTabs;

  const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";

  // refscrapofpackageitem.txt
  // maps RefPackageItemCodeName="PACKAGE_ITEM_ETC_HP_POTION_01" to item data
  //  1->1 (packages are unique, items might not be)
  const std::string kScrapOfPackageItemFilename = "refscrapofpackageitem.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing scrap of package item.";
  }
  auto scrapOfPackageItemPath = kTextdataDirectory + kScrapOfPackageItemFilename;
  auto scrapOfPackageItemStr = getFileDataAsString(pk2Reader, scrapOfPackageItemPath);
  parseDataFile<sro::pk2::ref::ScrapOfPackageItem>(scrapOfPackageItemStr, sro::pk2::parsing::isValidScrapOfPackageItemLine, sro::pk2::parsing::parseScrapOfPackageItemLine, [&scrapOfPackageItemMap](sro::pk2::ref::ScrapOfPackageItem &&package){
    scrapOfPackageItemMap.emplace(package.refPackageItemCodeName, std::move(package));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << scrapOfPackageItemMap.size() << " scrap of package item(s)";
  }

  // refshoptab.txt
  //  multimaps RefTabGroupCodeName="STORE_CA_POTION_GROUP1" to CodeName128="STORE_CA_POTION_TAB1"
  //  n->1 (tabs are unique)
  const std::string kShopTabFilename = "refshoptab.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing shop tab.";
  }
  auto shopTabPath = kTextdataDirectory + kShopTabFilename;
  auto shopTabStr = getFileDataAsString(pk2Reader, shopTabPath);
  parseDataFile<sro::pk2::ref::ShopTab>(shopTabStr, sro::pk2::parsing::isValidShopTabLine, sro::pk2::parsing::parseShopTabLine, [&shopTabs](sro::pk2::ref::ShopTab &&tab){
    shopTabs.emplace_back(std::move(tab));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << shopTabs.size() << " shop tab(s)";
  }

  // refshopgroup.txt
  //  multimaps RefNPCCodeName="NPC_CA_POTION" to CodeName128="GROUP_STORE_CA_POTION"
  //  n->1 (groups are unique)
  const std::string kShopGroupFilename = "refshopgroup.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing shop group.";
  }
  auto shopGroupPath = kTextdataDirectory + kShopGroupFilename;
  auto shopGroupStr = getFileDataAsString(pk2Reader, shopGroupPath);
  parseDataFile<sro::pk2::ref::ShopGroup>(shopGroupStr, sro::pk2::parsing::isValidShopGroupLine, sro::pk2::parsing::parseShopGroupLine, [&shopGroups](sro::pk2::ref::ShopGroup &&group){
    shopGroups.emplace_back(std::move(group));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << shopGroups.size() << " shop group(s)";
  }

  // refshopgoods.txt
  //  multimaps RefTabCodeName="STORE_CA_POTION_TAB1" to { RefPackageItemCodeName="PACKAGE_ITEM_ETC_HP_POTION_01", SlotIndex=0 }
  //  n->n
  const std::string kShopGoodsFilename = "refshopgoods.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing shop goods.";
  }
  auto shopGoodsPath = kTextdataDirectory + kShopGoodsFilename;
  auto shopGoodsStr = getFileDataAsString(pk2Reader, shopGoodsPath);
  parseDataFile<sro::pk2::ref::ShopGood>(shopGoodsStr, sro::pk2::parsing::isValidShopGoodLine, sro::pk2::parsing::parseShopGoodLine, [&shopGoods](sro::pk2::ref::ShopGood &&good){
    shopGoods.emplace_back(std::move(good));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << shopGoods.size() << " shop goods";
  }

  // refmappingshopgroup.txt
  //  maps RefShopGroupCodeName="GROUP_STORE_CA_POTION" to RefShopCodeName="STORE_CA_POTION"
  //  n->n
  const std::string kMappingShopGroupFilename = "refmappingshopgroup.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing mapping shop group.";
  }
  auto mappingShopGroupPath = kTextdataDirectory + kMappingShopGroupFilename;
  auto mappingShopGroupStr = getFileDataAsString(pk2Reader, mappingShopGroupPath);
  parseDataFile<sro::pk2::ref::MappingShopGroup>(mappingShopGroupStr, sro::pk2::parsing::isValidMappingShopGroupLine, sro::pk2::parsing::parseMappingShopGroupLine, [&mappingShopGroups](sro::pk2::ref::MappingShopGroup &&mapping){
    mappingShopGroups.emplace_back(std::move(mapping));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << mappingShopGroups.size() << " mapping shop group";
  }

  // refmappingshopwithtab.txt
  //  multimaps RefShopCodeName="STORE_CA_POTION" to RefTabGroupCodeName="STORE_CA_POTION_GROUP1"
  //  n->n
  const std::string kMappingShopWithTabFilename = "refmappingshopwithtab.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "Parsing mapping shop with tab.";
  }
  auto mappingShopWithTabPath = kTextdataDirectory + kMappingShopWithTabFilename;
  auto mappingShopWithTabStr = getFileDataAsString(pk2Reader, mappingShopWithTabPath);
  parseDataFile<sro::pk2::ref::MappingShopWithTab>(mappingShopWithTabStr, sro::pk2::parsing::isValidMappingShopWithTabLine, sro::pk2::parsing::parseMappingShopWithTabLine, [&mappingShopWithTabs](sro::pk2::ref::MappingShopWithTab &&mapping){
    mappingShopWithTabs.emplace_back(std::move(mapping));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    VLOG(1) << "  Cached " << mappingShopWithTabs.size() << " mapping shop with tab";
  }

  // Packages have already been created when we parsed everything
  // Create all the tabs (which will each contain packages)
  std::map<std::string, pk2::Tab> tabMap;
  for (const auto &shopGood : shopGoods) {
    // Make sure we've created an entry for this tab
    const auto &tabName = shopGood.refTabCodeName;
    if (tabMap.find(tabName) == tabMap.end()) {
      tabMap.emplace(tabName, pk2::Tab(tabName));
    }
    auto &tab = tabMap.at(tabName);
    // Find package
    auto it = scrapOfPackageItemMap.find(shopGood.refPackageItemCodeName);
    if (it ==  scrapOfPackageItemMap.end()) {
      throw std::runtime_error("Tab data references a package that we dont have");
    }
    // Add package to tab
    tab.addPackageAtSlot(it->second, shopGood.slotIndex);
  }

  // Add tabs for NPCs
  // For an NPC, get all unique tab groups
  for (const auto &shopGroup : shopGroups) {
    const auto &npc = shopGroup.refNPCCodeName;
    if (npc != "xxx") {
      // Ignoring this weird case. Seems to be for the item mall
      // First, find all unique tab groups for this NPC
      std::vector<std::string> tabGroups;
      const auto &group1 = shopGroup.codeName128;
      for (const auto &mappingShopGroup : mappingShopGroups) {
        if (mappingShopGroup.refShopGroupCodeName == group1)  {
          const auto &shop = mappingShopGroup.refShopCodeName;
          for (const auto &mappingShopWithTab : mappingShopWithTabs) {
            if (mappingShopWithTab.refShopCodeName == shop)  {
              const auto &tabGroup = mappingShopWithTab.refTabGroupCodeName;
              if (std::find(tabGroups.begin(), tabGroups.end(), tabGroup) == tabGroups.end()) {
                // First time finding this tab
                tabGroups.emplace_back(tabGroup);
              } else {
                VLOG(2) << "Found a duplicate tab group for NPC \"" << npc << "\"";
              }
            }
          }
        }
      }
      // Now, add each tab within each group to the NPC's shop

      uint8_t tabCount=0;
      for (const auto tabGroup : tabGroups) {
        for (const auto &shopTab : shopTabs) {
          if (shopTab.refTabGroupCodeName == tabGroup)  {
            const auto &tabName = shopTab.codeName128;
            auto tabIt = tabMap.find(tabName);
            if (tabIt != tabMap.end()) {
              auto &tab = tabIt->second;
              shopData_.addTabToNpc(npc, tabCount, tab);
              ++tabCount;
            } else {
              VLOG(2) << "Tab \"" << tabName << "\" didnt have any goods in it. Not adding to shop for any NPC";
            }
          }
        }
      }
    }
  }
}

void GameData::parseMagicOptionData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kMagicOptionDataPath = "server_dep\\silkroad\\textdata\\magicoption.txt";
  auto magicOptionStr = getFileDataAsString(pk2Reader, kMagicOptionDataPath);
  parseDataFile<sro::pk2::ref::MagicOption>(magicOptionStr, sro::pk2::parsing::isValidMagicOptionDataLine, sro::pk2::parsing::parseMagicOptionDataLine, std::bind(&MagicOptionData::addItem, &magicOptionData_, std::placeholders::_1));
}

void GameData::parseLevelData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kLevelDataPath = "server_dep\\silkroad\\textdata\\leveldata.txt";
  auto levelDataStr = getFileDataAsString(pk2Reader, kLevelDataPath);
  parseDataFile<sro::pk2::ref::Level>(levelDataStr, sro::pk2::parsing::isValidLevelDataLine, sro::pk2::parsing::parseLevelDataLine, std::bind(&LevelData::addLevelItem, &levelData_, std::placeholders::_1));
}

void GameData::parseRefRegion(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kRefRegionPath = "server_dep\\silkroad\\textdata\\refregion.txt";
  auto refRegionStr = getFileDataAsString(pk2Reader, kRefRegionPath);
  parseDataFile<sro::pk2::ref::Region>(refRegionStr, sro::pk2::parsing::isValidRefRegionLine, sro::pk2::parsing::parseRefRegionLine, std::bind(&RefRegion::addRegion, &refRegion_, std::placeholders::_1));
}

void GameData::parseTextData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  VLOG(3) << "Parsing Text Zone Name";
  parseTextZoneName(pk2Reader);
  VLOG(3) << "Parsing Text";
  parseText(pk2Reader);
  VLOG(3) << "Parsing Mastery Data";
  parseMasteryData(pk2Reader);
  VLOG(3) << "Parsing Text Ui System";
  parseTextUiSystem(pk2Reader);
}

void GameData::parseTextZoneName(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kTextZoneNamePath = "server_dep\\silkroad\\textdata\\textzonename.txt";
  auto textZoneNameStr = getFileDataAsString(pk2Reader, kTextZoneNamePath);
  parseDataFile<sro::pk2::ref::TextZoneName>(textZoneNameStr, sro::pk2::parsing::isValidTextDataLine, sro::pk2::parsing::parseTextZoneNameLine, std::bind(&TextZoneNameData::addItem, &textZoneNameData_, std::placeholders::_1));
}

void GameData::parseText(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterTextDataName = "textdataname.txt";
  const std::string kMasterTextDataPath = kTextdataDirectory + kMasterTextDataName;
  sro::pk2::PK2Entry masterTextDataEntry = pk2Reader.getEntry(kMasterTextDataPath);
  auto masterTextDataData = pk2Reader.getEntryData(masterTextDataEntry);
  auto textDataFilenames = sro::pk2::parsing::fileDataToStringLines(masterTextDataData);
  for (auto textDataFilename : textDataFilenames) {
    auto textDataPath = kTextdataDirectory + textDataFilename;
    VLOG(1) << absl::StreamFormat("Parsing file \"%s\"", textDataPath);
    sro::pk2::PK2Entry textEntry = pk2Reader.getEntry(textDataPath);
    auto textData = pk2Reader.getEntryData(textEntry);
    auto textLines = sro::pk2::parsing::fileDataToStringLines(textData);
    parseDataFile2<sro::pk2::ref::Text>(textLines, sro::pk2::parsing::isValidTextDataLine, sro::pk2::parsing::parseTextLine, std::bind(&TextData::addItem, &textData_, std::placeholders::_1));
  }
}

void GameData::parseMasteryData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kPath = "server_dep\\silkroad\\textdata\\skillmasterydata.txt";
  sro::pk2::PK2Entry entry = pk2Reader.getEntry(kPath);
  const auto data = pk2Reader.getEntryData(entry);
  const auto lines = sro::pk2::parsing::fileDataToStringLines(data);
  // TODO: Template on the add function instead
  parseDataFile2<sro::pk2::ref::Mastery>(lines, sro::pk2::parsing::isValidMasteryLine, sro::pk2::parsing::parseMasteryLine, std::bind(&MasteryData::addMastery, &masteryData_, std::placeholders::_1));
}

void GameData::parseTextUiSystem(sro::pk2::Pk2ReaderModern &pk2Reader) {
  const std::string kPath = "server_dep\\silkroad\\textdata\\textuisystem.txt";
  sro::pk2::PK2Entry entry = pk2Reader.getEntry(kPath);
  const auto data = pk2Reader.getEntryData(entry);
  const auto lines = sro::pk2::parsing::fileDataToStringLines(data);
  bool isMasteryNameSection{false};
  for (const std::string &line : lines) {
    if (line.size() >= 2 && line[0] == '/' && line[1] == '/') {
      // This line starts a new section.
      const std::vector<std::string_view> pieces = absl::StrSplit(line, '\t');
      if (pieces.size() > 0 && pieces[0] == "// Mastery Name") {
        isMasteryNameSection = true;
      } else {
        if (isMasteryNameSection) {
          // Done with Mastery Name section.
          // For now, there's nothing else we want in this file.
          return;
        }
      }
    } else {
      // Is not a section header.
      if (isMasteryNameSection) {
        // This is the section we care about.
        if (sro::pk2::parsing::isValidTextDataLine(line)) {
          textData_.addItem(sro::pk2::parsing::parseTextLine(line));
        } else {
          throw std::runtime_error(absl::StrFormat("When parsing textuisystem.txt for mastery names, encountered \"%s\" which is not a valid text data line", line));
        }
      }
    }
  }
}

void GameData::parseNavmeshData(sro::pk2::Pk2ReaderModern &pk2Reader) {
  VLOG(2) << "Parsing navmesh data";
  sro::pk2::NavmeshParser navmeshParser(pk2Reader);
  navmesh_ = navmeshParser.parseNavmesh();
  navmeshTriangulation_ = sro::navmesh::triangulation::NavmeshTriangulation(*navmesh_);
}

void GameData::parseRegionInfo(sro::pk2::Pk2ReaderModern &pk2Reader) {
  VLOG(2) << "Parsing region info";
  const std::string kRegionInfoEntryName = "regioninfo.txt";
  sro::pk2::PK2Entry regionInfoEntry = pk2Reader.getEntry(kRegionInfoEntryName);
  auto regionInfoData = pk2Reader.getEntryData(regionInfoEntry);
  regionInfo_ = sro::pk2::parsing::parseRegionInfo(regionInfoData);
}

} // namespace pk2