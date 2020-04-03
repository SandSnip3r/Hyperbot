#include "gameData.hpp"

#include "../../../common/pk2/pk2.h"
#include "../../../common/pk2/parsing/parsing.hpp"

#include <iostream>
#include <map>

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
  parseShopData(pk2Reader);
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

const ShopData& GameData::shopData() const {
  return shopData_;
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
      } catch (std::runtime_error &err) {
        std::cerr << "  Failed to parse character data \"" << line << "\"\n";
        std::cerr << "  " << err.what() << '\n';
      }
    }
  }
  std::cout << "  Cached " << characterData_.size() << " character(s)\n";
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
      } catch (std::runtime_error &err) {
        std::cerr << "  Failed to parse item data \"" << line << "\"\n";
        std::cerr << "  " << err.what() << '\n';
      }
    }
  }
  std::cout << "  Cached " << itemData_.size() << " item(s)\n";
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
      } catch (std::runtime_error &err) {
        std::cerr << "  Failed to parse skill data \"" << line << "\"\n";
        std::cerr << "  " << err.what() << '\n';
      }
    }
  }
  std::cout << "  Cached " << skillData_.size() << " skill(s)\n";
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
    } catch (std::runtime_error &err) {
      std::cerr << "  Failed to parse teleport data \"" << line << "\"\n";
      std::cerr << "  " << err.what() << '\n';
    }
  }
  std::cout << "  Cached " << teleportData_.size() << " teleport(s)\n";
}

struct Package {
  // Item data...
};

void GameData::parseShopData(Pk2ReaderModern &pk2Reader) {
  std::map<std::string, ref::ScrapOfPackageItem> scrapOfPackageItemMap;
  std::vector<ref::ShopTab> shopTabs;
  std::vector<ref::ShopGroup> shopGroups;
  std::vector<ref::ShopGood> shopGoods;
  std::vector<ref::MappingShopGroup> mappingShopGroups;
  std::vector<ref::MappingShopWithTab> mappingShopWithTabs;

	const std::string kTextdataDirectory = "server_dep\\silkroad\\textdata\\";

  // refscrapofpackageitem.txt
  // maps RefPackageItemCodeName="PACKAGE_ITEM_ETC_HP_POTION_01" to item data
  //  1->1 (packages are unique, items might not be)
  const std::string kScrapOfPackageItemFilename = "refscrapofpackageitem.txt";
  std::cout << "Parsing scrap of package item file \"" << kScrapOfPackageItemFilename << "\"\n";
  auto scrapOfPackageItemPath = kTextdataDirectory + kScrapOfPackageItemFilename;
  PK2Entry scrapOfPackageItemEntry = pk2Reader.getEntry(scrapOfPackageItemPath);
  auto scrapOfPackageItemData = pk2Reader.getEntryData(scrapOfPackageItemEntry);
  auto scrapOfPackageItemStr = parsing::fileDataToString(scrapOfPackageItemData);
  auto scrapOfPackageItemLines = parsing::split(scrapOfPackageItemStr, "\r\n");
  for (const auto &line : scrapOfPackageItemLines) {
    try {
      auto package = parsing::parseScrapOfPackageItemLine(line);
      scrapOfPackageItemMap.emplace(package.refPackageItemCodeName, package);
    } catch (std::runtime_error &err) {
      std::cerr << "  Failed to parse scrap of package item \"" << line << "\"\n";
      std::cerr << "  " << err.what() << '\n';
    }
  }
  std::cout << "  Cached " << scrapOfPackageItemMap.size() << " scrap of package item(s)\n";

  // refshoptab.txt
  //  multimaps RefTabGroupCodeName="STORE_CA_POTION_GROUP1" to CodeName128="STORE_CA_POTION_TAB1"
  //  n->1 (tabs are unique)
  const std::string kShopTabFilename = "refshoptab.txt";
  std::cout << "Parsing shop tab file \"" << kShopTabFilename << "\"\n";
  auto shopTabPath = kTextdataDirectory + kShopTabFilename;
  PK2Entry shopTabEntry = pk2Reader.getEntry(shopTabPath);
  auto shopTabData = pk2Reader.getEntryData(shopTabEntry);
  auto shopTabStr = parsing::fileDataToString(shopTabData);
  auto shopTabLines = parsing::split(shopTabStr, "\r\n");
  for (const auto &line : shopTabLines) {
    try {
      shopTabs.emplace_back(parsing::parseShopTabLine(line));
    } catch (std::runtime_error &err) {
      std::cerr << "  Failed to parse shop tab \"" << line << "\"\n";
      std::cerr << "  " << err.what() << '\n';
    }
  }
  std::cout << "  Cached " << shopTabs.size() << " shop tab(s)\n";

  // refshopgroup.txt
  //  multimaps RefNPCCodeName="NPC_CA_POTION" to CodeName128="GROUP_STORE_CA_POTION"
  //  n->1 (groups are unique)
  const std::string kShopGroupFilename = "refshopgroup.txt";
  std::cout << "Parsing shop group file \"" << kShopGroupFilename << "\"\n";
  auto shopGroupPath = kTextdataDirectory + kShopGroupFilename;
  PK2Entry shopGroupEntry = pk2Reader.getEntry(shopGroupPath);
  auto shopGroupData = pk2Reader.getEntryData(shopGroupEntry);
  auto shopGroupStr = parsing::fileDataToString(shopGroupData);
  auto shopGroupLines = parsing::split(shopGroupStr, "\r\n");
  for (const auto &line : shopGroupLines) {
    try {
      shopGroups.emplace_back(parsing::parseShopGroupLine(line));
    } catch (std::runtime_error &err) {
      std::cerr << "  Failed to parse shop group \"" << line << "\"\n";
      std::cerr << "  " << err.what() << '\n';
    }
  }
  std::cout << "  Cached " << shopGroups.size() << " shop group(s)\n";

  // refshopgoods.txt
  //  multimaps RefTabCodeName="STORE_CA_POTION_TAB1" to { RefPackageItemCodeName="PACKAGE_ITEM_ETC_HP_POTION_01", SlotIndex=0 }
  //  n->n
  const std::string kShopGoodsFilename = "refshopgoods.txt";
  std::cout << "Parsing shop goods file \"" << kShopGoodsFilename << "\"\n";
  auto shopGoodsPath = kTextdataDirectory + kShopGoodsFilename;
  PK2Entry shopGoodsEntry = pk2Reader.getEntry(shopGoodsPath);
  auto shopGoodsData = pk2Reader.getEntryData(shopGoodsEntry);
  auto shopGoodsStr = parsing::fileDataToString(shopGoodsData);
  auto shopGoodsLines = parsing::split(shopGoodsStr, "\r\n");
  for (const auto &line : shopGoodsLines) {
    try {
      shopGoods.emplace_back(parsing::parseShopGoodLine(line));
    } catch (std::runtime_error &err) {
      std::cerr << "  Failed to parse shop goods \"" << line << "\"\n";
      std::cerr << "  " << err.what() << '\n';
    }
  }
  std::cout << "  Cached " << shopGoods.size() << " shop goods\n";

  // refmappingshopgroup.txt
  //  maps RefShopGroupCodeName="GROUP_STORE_CA_POTION" to RefShopCodeName="STORE_CA_POTION"
  //  n->n
  const std::string kMappingShopGroupFilename = "refmappingshopgroup.txt";
  std::cout << "Parsing mapping shop group file \"" << kMappingShopGroupFilename << "\"\n";
  auto mappingShopGroupPath = kTextdataDirectory + kMappingShopGroupFilename;
  PK2Entry mappingShopGroupEntry = pk2Reader.getEntry(mappingShopGroupPath);
  auto mappingShopGroupData = pk2Reader.getEntryData(mappingShopGroupEntry);
  auto mappingShopGroupStr = parsing::fileDataToString(mappingShopGroupData);
  auto mappingShopGroupLines = parsing::split(mappingShopGroupStr, "\r\n");
  for (const auto &line : mappingShopGroupLines) {
    try {
      mappingShopGroups.emplace_back(parsing::parseMappingShopGroupLine(line));
    } catch (std::runtime_error &err) {
      std::cerr << "  Failed to parse mapping shop group \"" << line << "\"\n";
      std::cerr << "  " << err.what() << '\n';
    }
  }
  std::cout << "  Cached " << mappingShopGroups.size() << " mapping shop group\n";

  // refmappingshopwithtab.txt
  //  multimaps RefShopCodeName="STORE_CA_POTION" to RefTabGroupCodeName="STORE_CA_POTION_GROUP1"
  //  n->n
  const std::string kMappingShopWithTabFilename = "refmappingshopwithtab.txt";
  std::cout << "Parsing mapping shop with tab file \"" << kMappingShopWithTabFilename << "\"\n";
  auto mappingShopWithTabPath = kTextdataDirectory + kMappingShopWithTabFilename;
  PK2Entry mappingShopWithTabEntry = pk2Reader.getEntry(mappingShopWithTabPath);
  auto mappingShopWithTabData = pk2Reader.getEntryData(mappingShopWithTabEntry);
  auto mappingShopWithTabStr = parsing::fileDataToString(mappingShopWithTabData);
  auto mappingShopWithTabLines = parsing::split(mappingShopWithTabStr, "\r\n");
  for (const auto &line : mappingShopWithTabLines) {
    try {
      mappingShopWithTabs.emplace_back(parsing::parseMappingShopWithTabLine(line));
    } catch (std::runtime_error &err) {
      std::cerr << "  Failed to parse mapping shop with tab \"" << line << "\"\n";
      std::cerr << "  " << err.what() << '\n';
    }
  }
  std::cout << "  Cached " << mappingShopWithTabs.size() << " mapping shop with tab\n";

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

} // namespace pk2