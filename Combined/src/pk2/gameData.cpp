#include "gameData.hpp"

#include "logging.hpp"
#include "math/position.hpp"
#include "pk2/parsing/navmeshParser.hpp"
#include "../../../common/pk2/pk2.h"
#include "../../../common/pk2/parsing/parsing.hpp"

#include "math_helpers.h"

#include <functional>
#include <iostream>
#include <fstream>
#include <map>

namespace pk2 {

namespace fs = std::filesystem;

GameData::GameData(const fs::path &kSilkroadPath) : kSilkroadPath_(kSilkroadPath) {
  try {
    auto kDataPath = kSilkroadPath_ / "Data.pk2";
    Pk2ReaderModern pk2Reader{kDataPath};
    parseData(pk2Reader);
  } catch (std::exception &ex) {
    throw std::runtime_error(std::string("Failed to parse Data.Pk2 at path \""+kSilkroadPath_.string()+"\". Error: \"")+ex.what()+"\"");
  }
  try {
    auto kMediaPath = kSilkroadPath_ / "Media.pk2";
    Pk2ReaderModern pk2Reader{kMediaPath};
    parseMedia(pk2Reader);
  } catch (std::exception &ex) {
    throw std::runtime_error(std::string("Failed to parse Media.Pk2 at path \""+kSilkroadPath_.string()+"\". Error: \"")+ex.what()+"\"");
  }
}

void GameData::parseData(Pk2ReaderModern &pk2Reader) {
  parseNavmeshData(pk2Reader);
}

void GameData::parseMedia(Pk2ReaderModern &pk2Reader) {
  std::vector<std::thread> thrs;
  parseDivisionInfo(pk2Reader);
  parseShopData(pk2Reader);
  parseMagicOptionData(pk2Reader);
  thrs.emplace_back(&GameData::parseCharacterData, this, std::ref(pk2Reader));
  thrs.emplace_back(&GameData::parseItemData, this, std::ref(pk2Reader));
  thrs.emplace_back(&GameData::parseSkillData, this, std::ref(pk2Reader));
  parseTeleportData(pk2Reader);
  for (auto &thr : thrs) {
    thr.join();
  }
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

const ShopData& GameData::shopData() const {
  return shopData_;
}

const SkillData& GameData::skillData() const {
  return skillData_;
}

const MagicOptionData& GameData::magicOptionData() const {
  return magicOptionData_;
}

const TeleportData& GameData::teleportData() const {
  return teleportData_;
}

const navmesh::triangulation::NavmeshTriangulation& GameData::navmeshTriangulation() const {
  if (!navmeshTriangulation_.has_value()) {
    throw std::runtime_error("Asking for navmesh triangulation which does not exist");
  }
  return navmeshTriangulation_.value();
}

void GameData::parseDivisionInfo(Pk2ReaderModern &pk2Reader) {
  const std::string kDivisionInfoEntryName = "DIVISIONINFO.TXT";
  PK2Entry divisionInfoEntry = pk2Reader.getEntry(kDivisionInfoEntryName);
  auto divisionInfoData = pk2Reader.getEntryData(divisionInfoEntry);
  divisionInfo_ = parsing::parseDivisionInfo(divisionInfoData);
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
    std::cout << "One more\n";
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

} // namespace (anonymous)

void GameData::parseCharacterData(Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterCharacterdataName = "characterdata.txt";
  const std::string kMasterCharacterdataPath = kTextdataDirectory + kMasterCharacterdataName;
  PK2Entry masterCharacterdataEntry = pk2Reader.getEntry(kMasterCharacterdataPath);

  auto masterCharacterdataData = pk2Reader.getEntryData(masterCharacterdataEntry);
  // auto masterCharacterdataStr = parsing::fileDataToString(masterCharacterdataData);
  // auto characterdataFilenames = parsing::split(masterCharacterdataStr, "\r\n");
  auto characterdataFilenames = parsing::fileDataToStringLines(masterCharacterdataData);

  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing character data" << std::endl;
  }
  for (auto characterdataFilename : characterdataFilenames) {
    auto characterdataPath = kTextdataDirectory + characterdataFilename;
    PK2Entry characterdataEntry = pk2Reader.getEntry(characterdataPath);
    auto characterdataData = pk2Reader.getEntryData(characterdataEntry);
    // auto characterdataStr = parsing::fileDataToString(characterdataData);
    auto characterdataLines = parsing::fileDataToStringLines(characterdataData);
    try {
      parseDataFile2<ref::Character>(characterdataLines, parsing::isValidCharacterdataLine, parsing::parseCharacterdataLine, std::bind(&CharacterData::addCharacter, &characterData_, std::placeholders::_1));
      // parseDataFile<ref::Character>(characterdataStr, parsing::isValidCharacterdataLine, parsing::parseCharacterdataLine, std::bind(&CharacterData::addCharacter, &characterData_, std::placeholders::_1));
    } catch (std::exception &ex) {
      std::cout << "Exception while parsing character data\n";
      std::cout << "  " << ex.what() << '\n';
    }
  }
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << characterData_.size() << " character(s)" << std::endl;
  }
}

void GameData::parseItemData(Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMasterItemdataName = "itemdata.txt";
  const std::string kMasterItemdataPath = kTextdataDirectory + kMasterItemdataName;
  PK2Entry masterItemdataEntry = pk2Reader.getEntry(kMasterItemdataPath);

  auto masterItemdataData = pk2Reader.getEntryData(masterItemdataEntry);
  auto masterItemdataStr = parsing::fileDataToString(masterItemdataData);
  auto itemdataFilenames = parsing::split(masterItemdataStr, "\r\n");

  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing item data" << std::endl;
  }
  for (auto itemdataFilename : itemdataFilenames) {
    auto itemdataPath = kTextdataDirectory + itemdataFilename;
    PK2Entry itemdataEntry = pk2Reader.getEntry(itemdataPath);
    auto itemdataData = pk2Reader.getEntryData(itemdataEntry);
    auto itemdataStr = parsing::fileDataToString(itemdataData);
    parseDataFile<ref::Item>(itemdataStr, parsing::isValidItemdataLine, parsing::parseItemdataLine, std::bind(&ItemData::addItem, &itemData_, std::placeholders::_1));
  }
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << itemData_.size() << " item(s)" << std::endl;
  }
}

void GameData::parseSkillData(Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  // Prefer the encrypted file, as the client uses this and not the unencrypted version (skilldata.txt)
  const std::string kMasterSkilldataName = "skilldataenc.txt";
  const std::string kMasterSkilldataPath = kTextdataDirectory + kMasterSkilldataName;
  PK2Entry masterSkilldataEntry = pk2Reader.getEntry(kMasterSkilldataPath);

  auto masterSkilldataData = pk2Reader.getEntryData(masterSkilldataEntry);
  auto masterSkilldataStr = parsing::fileDataToString(masterSkilldataData);
  auto skilldataFilenames = parsing::split(masterSkilldataStr, "\r\n");

  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing skill data" << std::endl;
  }
  for (auto skilldataFilename : skilldataFilenames) {
    auto skilldataPath = kTextdataDirectory + skilldataFilename;
    PK2Entry skilldataEntry = pk2Reader.getEntry(skilldataPath);
    auto skilldataData = pk2Reader.getEntryData(skilldataEntry);
    // Decrypt this skill data
    parsing::decryptSkillData(skilldataData);
    auto skilldataStr = parsing::fileDataToString(skilldataData);
    parseDataFile<ref::Skill>(skilldataStr, parsing::isValidSkilldataLine, parsing::parseSkilldataLine, std::bind(&SkillData::addSkill, &skillData_, std::placeholders::_1));
  }
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << skillData_.size() << " skill(s)" << std::endl;
  }
}

void GameData::parseTeleportData(Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kTeleportDataFilename = "teleportbuilding.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing teleport data." << std::endl;
  }
  auto teleportDataPath = kTextdataDirectory + kTeleportDataFilename;
  PK2Entry teleportDataEntry = pk2Reader.getEntry(teleportDataPath);
  auto teleportDataData = pk2Reader.getEntryData(teleportDataEntry);
  auto teleportDataStr = parsing::fileDataToString(teleportDataData);
  parseDataFile<ref::Teleport>(teleportDataStr, parsing::isValidTeleportbuildingLine, parsing::parseTeleportbuildingLine, std::bind(&TeleportData::addTeleport, &teleportData_, std::placeholders::_1));
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << teleportData_.size() << " teleport(s)" << std::endl;
  }
}

void GameData::parseShopData(Pk2ReaderModern &pk2Reader) {
  // Maps Package name to item data
  std::map<std::string, ref::ScrapOfPackageItem> scrapOfPackageItemMap;
  // Maps a ShopTabGroup(I think an option within the NPC's dialog) to N tabs
  std::vector<ref::ShopTab> shopTabs;
  // List of NPC->ShopGroup
  std::vector<ref::ShopGroup> shopGroups;
  // List of Tab->Good
  std::vector<ref::ShopGood> shopGoods;
  // List of ShopGroup->Shop
  std::vector<ref::MappingShopGroup> mappingShopGroups;
  // List of Shop->ShopTabGroup
  std::vector<ref::MappingShopWithTab> mappingShopWithTabs;

	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";

  // refscrapofpackageitem.txt
  // maps RefPackageItemCodeName="PACKAGE_ITEM_ETC_HP_POTION_01" to item data
  //  1->1 (packages are unique, items might not be)
  const std::string kScrapOfPackageItemFilename = "refscrapofpackageitem.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing scrap of package item.";
  }
  auto scrapOfPackageItemPath = kTextdataDirectory + kScrapOfPackageItemFilename;
  PK2Entry scrapOfPackageItemEntry = pk2Reader.getEntry(scrapOfPackageItemPath);
  auto scrapOfPackageItemData = pk2Reader.getEntryData(scrapOfPackageItemEntry);
  auto scrapOfPackageItemStr = parsing::fileDataToString(scrapOfPackageItemData);
  parseDataFile<ref::ScrapOfPackageItem>(scrapOfPackageItemStr, parsing::isValidScrapOfPackageItemLine, parsing::parseScrapOfPackageItemLine, [&scrapOfPackageItemMap](ref::ScrapOfPackageItem &&package){
    scrapOfPackageItemMap.emplace(package.refPackageItemCodeName, std::move(package));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << scrapOfPackageItemMap.size() << " scrap of package item(s)\n";
  }

  // refshoptab.txt
  //  multimaps RefTabGroupCodeName="STORE_CA_POTION_GROUP1" to CodeName128="STORE_CA_POTION_TAB1"
  //  n->1 (tabs are unique)
  const std::string kShopTabFilename = "refshoptab.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing shop tab.";
  }
  auto shopTabPath = kTextdataDirectory + kShopTabFilename;
  PK2Entry shopTabEntry = pk2Reader.getEntry(shopTabPath);
  auto shopTabData = pk2Reader.getEntryData(shopTabEntry);
  auto shopTabStr = parsing::fileDataToString(shopTabData);
  parseDataFile<ref::ShopTab>(shopTabStr, parsing::isValidShopTabLine, parsing::parseShopTabLine, [&shopTabs](ref::ShopTab &&tab){
    shopTabs.emplace_back(std::move(tab));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << shopTabs.size() << " shop tab(s)\n";
  }

  // refshopgroup.txt
  //  multimaps RefNPCCodeName="NPC_CA_POTION" to CodeName128="GROUP_STORE_CA_POTION"
  //  n->1 (groups are unique)
  const std::string kShopGroupFilename = "refshopgroup.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing shop group.";
  }
  auto shopGroupPath = kTextdataDirectory + kShopGroupFilename;
  PK2Entry shopGroupEntry = pk2Reader.getEntry(shopGroupPath);
  auto shopGroupData = pk2Reader.getEntryData(shopGroupEntry);
  auto shopGroupStr = parsing::fileDataToString(shopGroupData);
  parseDataFile<ref::ShopGroup>(shopGroupStr, parsing::isValidShopGroupLine, parsing::parseShopGroupLine, [&shopGroups](ref::ShopGroup &&group){
    shopGroups.emplace_back(std::move(group));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << shopGroups.size() << " shop group(s)\n";
  }

  // refshopgoods.txt
  //  multimaps RefTabCodeName="STORE_CA_POTION_TAB1" to { RefPackageItemCodeName="PACKAGE_ITEM_ETC_HP_POTION_01", SlotIndex=0 }
  //  n->n
  const std::string kShopGoodsFilename = "refshopgoods.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing shop goods.";
  }
  auto shopGoodsPath = kTextdataDirectory + kShopGoodsFilename;
  PK2Entry shopGoodsEntry = pk2Reader.getEntry(shopGoodsPath);
  auto shopGoodsData = pk2Reader.getEntryData(shopGoodsEntry);
  auto shopGoodsStr = parsing::fileDataToString(shopGoodsData);
  parseDataFile<ref::ShopGood>(shopGoodsStr, parsing::isValidShopGoodLine, parsing::parseShopGoodLine, [&shopGoods](ref::ShopGood &&good){
    shopGoods.emplace_back(std::move(good));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << shopGoods.size() << " shop goods\n";
  }

  // refmappingshopgroup.txt
  //  maps RefShopGroupCodeName="GROUP_STORE_CA_POTION" to RefShopCodeName="STORE_CA_POTION"
  //  n->n
  const std::string kMappingShopGroupFilename = "refmappingshopgroup.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing mapping shop group.";
  }
  auto mappingShopGroupPath = kTextdataDirectory + kMappingShopGroupFilename;
  PK2Entry mappingShopGroupEntry = pk2Reader.getEntry(mappingShopGroupPath);
  auto mappingShopGroupData = pk2Reader.getEntryData(mappingShopGroupEntry);
  auto mappingShopGroupStr = parsing::fileDataToString(mappingShopGroupData);
  parseDataFile<ref::MappingShopGroup>(mappingShopGroupStr, parsing::isValidMappingShopGroupLine, parsing::parseMappingShopGroupLine, [&mappingShopGroups](ref::MappingShopGroup &&mapping){
    mappingShopGroups.emplace_back(std::move(mapping));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << mappingShopGroups.size() << " mapping shop group\n";
  }

  // refmappingshopwithtab.txt
  //  multimaps RefShopCodeName="STORE_CA_POTION" to RefTabGroupCodeName="STORE_CA_POTION_GROUP1"
  //  n->n
  const std::string kMappingShopWithTabFilename = "refmappingshopwithtab.txt";
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "Parsing mapping shop with tab.";
  }
  auto mappingShopWithTabPath = kTextdataDirectory + kMappingShopWithTabFilename;
  PK2Entry mappingShopWithTabEntry = pk2Reader.getEntry(mappingShopWithTabPath);
  auto mappingShopWithTabData = pk2Reader.getEntryData(mappingShopWithTabEntry);
  auto mappingShopWithTabStr = parsing::fileDataToString(mappingShopWithTabData);
  parseDataFile<ref::MappingShopWithTab>(mappingShopWithTabStr, parsing::isValidMappingShopWithTabLine, parsing::parseMappingShopWithTabLine, [&mappingShopWithTabs](ref::MappingShopWithTab &&mapping){
    mappingShopWithTabs.emplace_back(std::move(mapping));
  });
  {
    std::unique_lock<std::mutex> lock(printMutex_);
    std::cout << "  Cached " << mappingShopWithTabs.size() << " mapping shop with tab\n";
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
                std::cout << "Found a duplicate tab group for NPC \"" << npc << "\"\n";
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
              std::cout << "Tab \"" << tabName << "\" didnt have any goods in it. Not adding to shop for any NPC\n";
            }
          }
        }
      }
    }
  }
}

void GameData::parseMagicOptionData(Pk2ReaderModern &pk2Reader) {
	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";
  const std::string kMagicOptionDataFilename = "magicoption.txt";
  const std::string kMagicOptionDataPath = kTextdataDirectory + kMagicOptionDataFilename;
  PK2Entry magicOptionDataEntry = pk2Reader.getEntry(kMagicOptionDataPath);
  auto magicOptionData = pk2Reader.getEntryData(magicOptionDataEntry);
  auto magicOptionStr = parsing::fileDataToString(magicOptionData);
  parseDataFile<ref::MagicOption>(magicOptionStr, parsing::isValidMagicOptionDataLine, parsing::parseMagicOptionDataLine, std::bind(&MagicOptionData::addItem, &magicOptionData_, std::placeholders::_1));
}

void GameData::parseNavmeshData(Pk2ReaderModern &pk2Reader) {
  std::cout << "Parsing navmesh data\n";
  // pk2::parsing::NavmeshParser navmeshParser(pk2Reader);
  // navmesh_ = navmeshParser.parseNavmesh();
  // navmeshTriangulation_ = navmesh::triangulation::NavmeshTriangulation(*navmesh_);

  // ==============OLD Below?==============
  // for (int regionX=0; regionX<255; ++regionX) {
  //   for (int regionY=0; regionY<128; ++regionY) {
  //     const auto regionId = math::position::worldRegionIdFromXY(regionX, regionY);
  //     // const auto regionId = math::position::worldRegionIdFromXY(70, 107);
  //     const auto [x,y] = math::position::regionXYFromRegionId(regionId);
  //     // std::cout << "Region " << regionId << " is " << x << ',' << y << std::endl;
  //     if (navmeshParser.regionIsEnabled(regionId)) {
  //       const auto regionNavmeshData = navmeshParser.parseRegionNavmesh(regionId);
  //       auto regionNavmesh = navmesh::buildNavmeshForRegion(regionNavmeshData, navmeshParser, false);
  //       regionNavmeshes_.emplace(regionId, std::move(regionNavmesh));
  //     }
  //   }
  // }
}

// const pathfinder::navmesh::AStarNavmeshInterface& GameData::getNavmeshForRegionId(const uint16_t regionId) const {
//   const auto it = regionNavmeshes_.find(regionId);
//   if (it == regionNavmeshes_.end()) {
//     throw std::runtime_error("GameData::getNavmeshForRegionId: Asking for a navmesh for a region which does not exist");
//   }
//   if (!it->second) {
//     throw std::runtime_error("GameData::getNavmeshForRegionId: Asking for a navmesh that is nullptr");
//   }
//   return *(it->second.get());
// }

} // namespace pk2