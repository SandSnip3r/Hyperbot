#include "gameData.hpp"

#include "../math/matrix.hpp"
#include "../math/position.hpp"
#include "../../../common/pk2/pk2.h"
#include "../../../common/pk2/parsing/parsing.hpp"

#include "behaviorBuilder.h"
#include "math_helpers.h"
#include "triangle/triangle_api.h"
#include "triangle_lib_navmesh.h"

#include <functional>
#include <iostream>
#include <fstream>
#include <map>

namespace navmesh_geometry_helpers {

ObjectResource transformObject(const ObjectResource &obj, const MapObjInfo &objInfo);
bool lineTrim(Vector &p1, Vector &p2);

} // namespace navmesh_geometry_helpers

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

const TeleportData& GameData::teleportData() const {
  return teleportData_;
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

void dumpPolyFile(const std::string &filename, const GameData::PointListType &inVertices, const GameData::EdgeListType &inEdges) {
  std::ofstream polyFile(filename);
  if (!polyFile) {
    std::cout << "Unable to open polyfile \"" << filename << "\" for writing\n";
    return;
  }

  // Vertices
  polyFile << inVertices.size() << " 2 0 0\n";
  for (int i=0; i<inVertices.size(); ++i) {
    const auto &vertex = inVertices.at(i);
    polyFile << i << ' ' << std::fixed << std::setprecision(12) << vertex.x() << ' ' << std::fixed << std::setprecision(12) << vertex.y() << '\n';
  }

  // Edges
  polyFile << inEdges.size() << " 1\n";
  for (int i=0; i<inEdges.size(); ++i) {
    const auto &edge = inEdges.at(i);
    polyFile << i << ' ' << edge.vertex0 << ' ' << edge.vertex1 << ' ' << edge.marker << '\n';
  }

  // Holes
  polyFile << "0\n";
  std::cout << "File \"" << filename << "\" written\n";
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

std::unique_ptr<pathfinder::navmesh::AStarNavmeshInterface> GameData::buildNavmeshForRegion(const RegionNavmesh &regionNavmesh, const NavmeshParser &navmeshParser, const bool createDebugPolyFile) {
  // ============================Lambdas============================
  // Now, extract data from the navmesh
  auto addVertexAndGetIndex = [](const Vector &p, PointListType &points) -> size_t {
    // TODO: Maybe give a tiny range for precision error
    auto it = std::find_if(points.begin(), points.end(), [&p](const pathfinder::Vector &otherPoint){
      return (pathfinder::math::equal(otherPoint.x(), p.x) && pathfinder::math::equal(otherPoint.y(), p.z));
    });
    if (it != points.end()) {
      // Point already exists in list
      // std::cout << "Vertex already exists! " << p.x << ',' << p.z << " at index " << std::distance(points.begin(), it) << '\n';
      return std::distance(points.begin(), it);
    }
    // std::cout << "Insert vertex " << p.x << ',' << p.z << " at index " << points.size() << '\n';
    points.emplace_back(p.x, p.z);
    return points.size()-1;
  };

  auto addEdge = [&addVertexAndGetIndex](Vector v1, Vector v2, PointListType &points, EdgeListType &edges) {
    const auto v1Index = addVertexAndGetIndex(v1, points);
    const auto v2Index = addVertexAndGetIndex(v2, points);
    if (std::find_if(edges.begin(), edges.end(), [v1Index, v2Index](const EdgeType &edge) {
      return (edge.vertex0 == v1Index && edge.vertex1 == v2Index) || (edge.vertex0 == v2Index && edge.vertex1 == v1Index);
    }) == edges.end()) {
      // std::cout << "Adding edge " << points.at(v1Index).x() << ',' << points.at(v1Index).y() << " -> " << points.at(v2Index).x() << ',' << points.at(v2Index).y() << std::endl;
      edges.emplace_back(v1Index, v2Index);
    } else {
      // std::cout << "Edge already exists " << points.at(v1Index).x() << ',' << points.at(v1Index).y() << " -> " << points.at(v2Index).x() << ',' << points.at(v2Index).y() << std::endl;
    }
  };

  auto addObjEdgeWithTrim = [&addVertexAndGetIndex, &addEdge](Vector v1, Vector v2, PointListType &points, EdgeListType &edges) {
    bool res = navmesh_geometry_helpers::lineTrim(v1, v2);
    if (res) {
      addEdge(v1, v2, points, edges);
    }
  };

  // =============================Data==============================
  PointListType inVertices;
  EdgeListType inEdges;

  // enum EdgeFlag : byte
  // {
  //     None = 0,
  //     BlockDst2Src = 1,
  //     BlockSrc2Dst = 2,
  //     Blocked = BlockDst2Src | BlockSrc2Dst,
  //     Internal = 4,
  //     Global = 8,
  //     Bridge = 16,
  //     Entrance = 32,  // Dungeon
  //     Bit6 = 64,
  //     Siege = 128,    // Fortress War (projectile passthrough)
  // }

  // // Lets just statically add the 4 edges of the entire region
  // // TODO: This doesnt really make sense, it's more useful to add to global edges of the region as they exist (except maybe modified by merging edges of the same type?)
  // addEdge(pathfinder::Vector{0,0,0}, pathfinder::Vector{1920,0,0}, inVertices, inEdges);
  // addEdge(pathfinder::Vector{1920,0,0}, pathfinder::Vector{1920,0,1920}, inVertices, inEdges);
  // addEdge(pathfinder::Vector{1920,0,1920}, pathfinder::Vector{0,0,1920}, inVertices, inEdges);
  // addEdge(pathfinder::Vector{0,0,1920}, pathfinder::Vector{0,0,0}, inVertices, inEdges);

  // It is possible that a region has no global edges nor instance edges, though it could be a walkable region.
  // Add the 4 corners as a minimum so there is a walkable space
  (void)addVertexAndGetIndex(Vector{0,0,0}, inVertices);
  (void)addVertexAndGetIndex(Vector{1920,0,0}, inVertices);
  (void)addVertexAndGetIndex(Vector{1920,0,1920}, inVertices);
  (void)addVertexAndGetIndex(Vector{0,0,1920}, inVertices);

  // Want all global edges
  for (const auto &edge : regionNavmesh.globalEdges) {
    // std::cout << "Global " << edge.min.x << ',' << edge.min.z << " -> " << edge.max.x << ',' << edge.max.z << '\n';
    addEdge(edge.min, edge.max, inVertices, inEdges);
  }

  // Only want constraining edges of interior edges
  for (const auto &edge : regionNavmesh.internalEdges) {
    if (((edge.flag & 1) ||
        (edge.flag & 2) ||
        (edge.flag & 16) ||
        (edge.flag & 128))/*  &&
        (edge.flag & 4) */) {
      // std::cout << "Internal " << edge.min.x << ',' << edge.min.z << " -> " << edge.max.x << ',' << edge.max.z << '\n';
      addEdge(edge.min, edge.max, inVertices, inEdges);
    } else {
    //   std::cout << "flag " << (int)edge.flag << '\n';
    }
  }

  // Get contstraining edges for object
  for (const auto &objectInstance : regionNavmesh.mapObjInfos) {
    auto it = navmeshParser.getObjectResourceMap().find(objectInstance.objectId);
    if (it == navmeshParser.getObjectResourceMap().end()) {
      std::cout << "Wait, cant find object " << objectInstance.objectId << '\n';
      continue;
    }
    const auto &objectResource = it->second;

    // Transform object
    const auto tranformedObjectResource = navmesh_geometry_helpers::transformObject(objectResource, objectInstance);

    for (const auto &cell : tranformedObjectResource.cells) {
      if (cell.eventZoneData) {
        // std::cout << "Found a cell with eventZoneData\n";
        // std::cout << "  " << (int)*(cell.eventZoneData) << '\n';
        addObjEdgeWithTrim(tranformedObjectResource.vertices.at(cell.vertex0),
                           tranformedObjectResource.vertices.at(cell.vertex1),
                           inVertices,
                           inEdges);
        addObjEdgeWithTrim(tranformedObjectResource.vertices.at(cell.vertex1),
                           tranformedObjectResource.vertices.at(cell.vertex2),
                           inVertices,
                           inEdges);
        addObjEdgeWithTrim(tranformedObjectResource.vertices.at(cell.vertex2),
                           tranformedObjectResource.vertices.at(cell.vertex0),
                           inVertices,
                           inEdges);
      }
    }

    for (const auto &edge : tranformedObjectResource.outlineEdges) {
      // uint16_t srcVertex, destVertex, srcCell, destCell;
      // if (((edge.flag & 1) ||
      //     (edge.flag & 2) ||
      //     (edge.flag & 16) ||
      //     (edge.flag & 128)) ||
      //     edge.eventZoneData) {
        addObjEdgeWithTrim(tranformedObjectResource.vertices.at(edge.srcVertex),
                           tranformedObjectResource.vertices.at(edge.destVertex),
                           inVertices,
                           inEdges);
      // }
    }

    for (const auto &edge : tranformedObjectResource.inlineEdges) {
      // uint16_t srcVertex, destVertex, srcCell, destCell;
      if (((edge.flag & 1) ||
          (edge.flag & 2) ||
          (edge.flag & 16) ||
          (edge.flag & 128)) ||
          edge.eventZoneData) {
        addObjEdgeWithTrim(tranformedObjectResource.vertices.at(edge.srcVertex),
                           tranformedObjectResource.vertices.at(edge.destVertex),
                           inVertices,
                           inEdges);
      }
    }
  }

  // TODO: Temporary aid for visualization; remove
  if (createDebugPolyFile) {
    dumpPolyFile("regionNavmesh.poly", inVertices, inEdges);
  }

  // ===============================================================================
  // =========================Ok, input data is transformed=========================
  // ===============================================================================

  // Some data init
	triangle::context *ctx;
	triangle::triangleio inputStruct;
	triangle::triangle_initialize_triangleio(&inputStruct);

	ctx = triangle::triangle_context_create();
  *(ctx->b) = pathfinder::BehaviorBuilder{}.getBehavior();

  // fill input structure vertices
  inputStruct.numberofpoints = inVertices.size();
  inputStruct.numberofpointattributes = 0;
	inputStruct.pointlist = (TRIANGLE_MACRO_REAL *) malloc((unsigned int) (2 * inVertices.size() * sizeof(TRIANGLE_MACRO_REAL)));
  int vertexIndex = 0;
  for (const auto &vertex : inVertices) {
    inputStruct.pointlist[2*vertexIndex] = vertex.x();
    inputStruct.pointlist[2*vertexIndex + 1] = vertex.y();
    ++vertexIndex;
  }

  // fill input structure edges
	inputStruct.numberofsegments = inEdges.size();
  inputStruct.segmentlist = (int *) malloc((unsigned int) (2 * inEdges.size() * sizeof(int)));
  inputStruct.segmentmarkerlist = (int *) malloc((unsigned int) (inEdges.size() * sizeof(int)));
  int edgeIndex = 0;
  for (const auto &edge : inEdges) {
    inputStruct.segmentlist[2*edgeIndex] = edge.vertex0;
    inputStruct.segmentlist[2*edgeIndex + 1] = edge.vertex1;
    inputStruct.segmentmarkerlist[edgeIndex] = edge.marker;
    ++edgeIndex;
  }

  // fill input structure (regions?)
  inputStruct.numberofregions = 0;

  // generate mesh
  auto beforeTime = std::chrono::high_resolution_clock::now();
  int meshCreateResult = triangle_mesh_create(ctx, &inputStruct);
  if (meshCreateResult < 0) {
    triangle::triangle_free_triangleio(&inputStruct);
    triangle::triangle_context_destroy(ctx);
    throw std::runtime_error("Error creating mesh "+std::to_string(meshCreateResult));
  }
  auto afterTime = std::chrono::high_resolution_clock::now();

  // write_nodes(ctx, "test-out.node");
  // write_edges(ctx, "test-out.edge");

  // Prepare data structures
  triangle::triangleio triangleData, triangleVoronoiData;
	triangle::triangle_initialize_triangleio(&triangleData);
	triangle::triangle_initialize_triangleio(&triangleVoronoiData);

  // Extract data from the context
  beforeTime = std::chrono::high_resolution_clock::now();
  int copyResult = triangle_mesh_copy(ctx, &triangleData, 1, 1, &triangleVoronoiData);
  afterTime = std::chrono::high_resolution_clock::now();
  if (copyResult < 0) {
    triangle::triangle_free_triangleio(&triangleData);
    triangle::triangle_free_triangleio(&triangleVoronoiData);
    triangle::triangle_free_triangleio(&inputStruct);
    triangle::triangle_context_destroy(ctx);
    throw std::runtime_error("Error copying data");
  }

  std::unique_ptr<pathfinder::navmesh::AStarNavmeshInterface> navmesh(new pathfinder::navmesh::TriangleLibNavmesh(triangleData, triangleVoronoiData));

  // Cleanup
  triangle::triangle_free_triangleio(&triangleData);
  triangle::triangle_free_triangleio(&triangleVoronoiData);
  triangle::triangle_free_triangleio(&inputStruct);
  triangle::triangle_context_destroy(ctx);

  return navmesh;
}

void GameData::parseNavmeshData(Pk2ReaderModern &pk2Reader) {
  std::cout << "Parsing navmesh data\n";
  NavmeshParser navmeshParser(pk2Reader);
  uint64_t parsingTimeTotal{0};
  uint64_t buildingTimeTotal{0};
  for (int regionX=0; regionX<255; ++regionX) {
    for (int regionY=0; regionY<128; ++regionY) {
      const auto regionId = math::position::worldRegionIdFromXY(regionX, regionY);
      // const auto regionId = math::position::worldRegionIdFromXY(70, 107);
      const auto [x,y] = math::position::regionXYFromRegionId(regionId);
      // std::cout << "Region " << regionId << " is " << x << ',' << y << std::endl;
      if (navmeshParser.regionIsEnabled(regionId)) {
        auto startTime = std::chrono::high_resolution_clock::now();
        const auto regionNavmeshData = navmeshParser.parseRegionNavmesh(regionId);
        auto endTime = std::chrono::high_resolution_clock::now();
        // std::cout << "parseRegionNavmesh took " << std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count() << " microseconds\n";
        parsingTimeTotal += std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count();
        startTime = std::chrono::high_resolution_clock::now();
        auto navmesh = buildNavmeshForRegion(regionNavmeshData, navmeshParser, false);
        endTime = std::chrono::high_resolution_clock::now();
        // std::cout << "buildNavmeshForRegion took " << std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count() << " microseconds\n";
        buildingTimeTotal += std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count();
        regionNavmeshes_.emplace(regionId, std::move(navmesh));
      }
    }
  }
  std::cout << "parsingTimeTotal: " << parsingTimeTotal << std::endl;
  std::cout << "buildingTimeTotal: " << buildingTimeTotal << std::endl;
  // Load region at East Jangan gate
  // const int kRegionX = 170;
  // const int kRegionY = 97;
  // Load region at Southeast corner of Jangan
  // const int kRegionX = 169;
  // const int kRegionY = 97;
  // Load region ____
  // navmesh_ = std::unique_ptr<pathfinder::navmesh::NavmeshInterface>(new pathfinder::navmesh::TriangleLibNavmesh(triangleData, triangleVoronoiData));
  // const int kRegionX = 69;
  // const int kRegionY = 101;
  // const uint16_t kRegionId = kRegionX&0xFF | ((kRegionY&0xFF)<<8);
  // if (navmeshParser.regionIsEnabled(kRegionId)) {
  //   std::cout << "Parsing region " << kRegionId << '\n';
  //   const auto regionNavmesh = navmeshParser.parseRegionNavmesh(kRegionId);
  //   const auto navmesh = buildNavmeshForRegion(regionNavmesh, navmeshParser);
  // } else {
  //   std::cout << "Not parsing region because it is disabled\n";
  // }
}

const pathfinder::navmesh::AStarNavmeshInterface& GameData::getNavmeshForRegionId(const uint16_t regionId) const {
  const auto it = regionNavmeshes_.find(regionId);
  if (it == regionNavmeshes_.end()) {
    throw std::runtime_error("GameData::getNavmeshForRegionId: Asking for a navmesh for a region which does not exist");
  }
  if (!it->second) {
    throw std::runtime_error("GameData::getNavmeshForRegionId: Asking for a navmesh that is nullptr");
  }
  return *(it->second.get());
}

} // namespace pk2

namespace navmesh_geometry_helpers {

bool lineTrim(Vector &p1, Vector &p2) {
  // std::cout << "Checking {(" << p1.x << ',' << p1.y << "),(" << p2.x << ',' << p2.y << ")}\n";
  // Compare this line against all boundaries of the region
  std::vector<std::pair<Vector,Vector>> boundaries = {{Vector(0,0,0), Vector(0,0,1920)},
                                                      {Vector(0,0,1920), Vector(1920,0,1920)},
                                                      {Vector(1920,0,1920), Vector(1920,0,0)},
                                                      {Vector(1920,0,0), Vector(0,0,0)}};
  for (const auto &boundary : boundaries) {
    // std::cout << "  against {(" << boundary.first.x << ',' << boundary.first.z << "),(" << boundary.second.x << ',' << boundary.second.z << ")}\n";
    // Check if lines intersect
    float &x1 = p1.x;
    float &y1 = p1.z;
    float &x2 = p2.x;
    float &y2 = p2.z;
    const float &x3 = boundary.first.x;
    const float &y3 = boundary.first.z;
    const float &x4 = boundary.second.x;
    const float &y4 = boundary.second.z;

    auto det = [](float a, float b, float c, float d) {
      return a*d-b*c;
    };

    float tNumerator = det(x1-x3, x3-x4, y1-y3, y3-y4);
    float uNumerator = det(x1-x2, x1-x3, y1-y2, y1-y3);
    float denom = det(x1-x2, x3-x4, y1-y2, y3-y4);

    if (denom == 0) {
      continue;
    }

    float t = tNumerator/denom;
    float u = -uNumerator/denom;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
      // Intersection lies on both segments
      // std::cout << "Line {(" << x1 << ',' << y1 << "),(" << x2 << ',' << y2 << ")} intersects with {(" << x3 << ',' << y3 << "),(" << x4 << ',' << y4 << ")}\n";

      // Trim the line based on the line it intersected with
      float intersectionX = x1+t*(x2-x1);
      float intersectionY = y1+t*(y2-y1);
      // std::cout << "Intersection at (" << intersectionX << ',' << intersectionY << ")\n";

      bool trimTheFirstPoint = false;
      if (x3 == x4) {
        // Vertical lines
        if (y3 == 0) {
          // Left side
          // std::cout << "Extends left";
          if (x1 < 0) {
            // std::cout << " trimming first point";
            trimTheFirstPoint = true;
          } else {
            // std::cout << " trimming second point";
            trimTheFirstPoint = false;
          }
        } else {
          // Right side
          // std::cout << "Extends right";
          if (x1 > 1920) {
            // std::cout << " trimming first point";
            trimTheFirstPoint = true;
          } else {
            // std::cout << " trimming second point";
            trimTheFirstPoint = false;
          }
        }
      } else {
        if (x3 == 0) {
          // Top side (y=1920)
          // std::cout << "Extends up";
          if (y1 > 1920) {
            // std::cout << " trimming first point";
            trimTheFirstPoint = true;
          } else {
            // std::cout << " trimming second point";
            trimTheFirstPoint = false;
          }
        } else {
          // Bottom side
          // std::cout << "Extends down";
          if (y1 < 0) {
            // std::cout << " trimming first point";
            trimTheFirstPoint = true;
          } else {
            // std::cout << " trimming second point";
            trimTheFirstPoint = false;
          }
        }
      }
      // std::cout << '\n';
      if (trimTheFirstPoint) {
        x1 = intersectionX;
        y1 = intersectionY;
        if (fabs(x1) < 1e-5) {
          x1 = 0;
        }
        if (fabs(y1) < 1e-5) {
          y1 = 0;
        }
        if (fabs(x1-1920) < 1e-5) {
          x1 = 1920;
        }
        if (fabs(y1-1920) < 1e-5) {
          y1 = 1920;
        }
      } else {
        x2 = intersectionX;
        y2 = intersectionY;
        if (fabs(x2) < 1e-5) {
          x2 = 0;
        }
        if (fabs(y2) < 1e-5) {
          y2 = 0;
        }
        if (fabs(x2-1920) < 1e-5) {
          x2 = 1920;
        }
        if (fabs(y2-1920) < 1e-5) {
          y2 = 1920;
        }
      }
      // std::cout << "Updated line to {(" << x1 << ',' << y1 << "),(" << x2 << ',' << y2 << ")}\n";

      // Update line segment to exclude the part that isnt inside the region
      // std::cout << "t:" << t << ", u:" << u << '\n';
      // std::cout << std::endl;
    } else {
      // std::cout << t << ',' << u << '\n';
    }
  }

  // If any point is outside of the region, then the entire line must be outside, return false
  bool pointOutside = false;
  pointOutside |= (p1.x < 0 || p1.x > 1920);
  pointOutside |= (p1.z < 0 || p1.z > 1920);
  pointOutside |= (p2.x < 0 || p2.x > 1920);
  pointOutside |= (p2.z < 0 || p2.z > 1920);
  // if (pointOutside) {
  //   std::cout << "line discarded " << p1.x << ',' << p1.z << " - " << p2.x << ',' << p2.z << '\n';
  // }
  return !pointOutside;
  // std::cout << "Trimed line {(" << p1.x << ',' << p1.z << "),(" << p2.x << ',' << p2.z << ")}\n";
}

ObjectResource transformObject(const ObjectResource &obj, const MapObjInfo &objInfo) {
  ObjectResource transformedObject = obj;

  Matrix4x4 rotationMatrix;
  rotationMatrix.setRotation(-objInfo.yaw, {0,1,0});
  Matrix4x4 translationMatrix;
  translationMatrix.setTranslation(objInfo.center);
  const Matrix4x4 transformationMatrix = translationMatrix*rotationMatrix;

  for (auto &vertex : transformedObject.vertices) {
    vertex = transformationMatrix*vertex;
  }

  return transformedObject;
}

} // namespace navmesh_geometry_helpers