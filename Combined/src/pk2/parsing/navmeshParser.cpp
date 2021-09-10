#include "navmeshParser.hpp"

#include "../../math/position.hpp"

#include <array>
#include <exception>
#include <fstream>
#include <set>
#include <string>
#include <sstream>

#include <iostream>

// ========================================================================
// =========================Basic helper functions=========================
// ========================================================================

template<typename T>
inline void parse(std::istringstream &inFile, T &data) {
  inFile.read(reinterpret_cast<char*>(&data), sizeof(T));
}

// ========================================================================
// =========================NavmeshParser functions========================
// ========================================================================

namespace pk2::parsing {

NavmeshParser::NavmeshParser(pk2::Pk2ReaderModern &pk2Reader) : pk2Reader_(pk2Reader) {
}

navmesh::Navmesh NavmeshParser::parseNavmesh() { 
  buildObjectFileInfoMap();
  parseMapInfo();

  // ======================Begin Threaded Version======================
  /*
  navmesh::Navmesh navmesh;
  std::vector<std::vector<std::function<void()>>> taskQueues;
  int taskQueueIndex=0;
  taskQueues.resize(std::thread::hardware_concurrency());
  for (int regionRow=0; regionRow<mapInfo_.mapHeight; ++regionRow) {
    for (int regionCol=0; regionCol<mapInfo_.mapWidth; ++regionCol) {
      const auto regionId = math::position::worldRegionIdFromXY(regionCol, regionRow);
      if (regionIsEnabled(regionId)) {
        taskQueues[taskQueueIndex].emplace_back(std::bind(&NavmeshParser::parseRegion, this, regionId, std::ref(navmesh)));
        ++taskQueueIndex;
        if (taskQueueIndex >= taskQueues.size())  {
          taskQueueIndex = 0;
        }
        // parseRegion(regionId, navmesh);
      }
    }
  }
  std::cout << "There are " << taskQueues.size() << " task queues with [ ";
  for (const auto &i : taskQueues) {
    std::cout << i.size() << ' ';
  }
  std::cout << "] items" << std::endl;
  std::vector<std::thread> threads;
  for (const auto &taskQueue : taskQueues) {
    threads.emplace_back([&taskQueue](){
      for (const auto &func : taskQueue) {
        func();
      }
    });
  }
  for (auto &thr : threads) {
    thr.join();
  }
  */
  // ====================== End Threaded Version ======================

  navmesh::Navmesh navmesh;
  for (int regionRow=0; regionRow<mapInfo_.mapHeight; ++regionRow) {
    for (int regionCol=0; regionCol<mapInfo_.mapWidth; ++regionCol) {
      const auto regionId = math::position::worldRegionIdFromXY(regionCol, regionRow);
      if (regionIsEnabled(regionId)) {
        parseRegion(regionId, navmesh);
      }
    }
  }

  navmesh.sanityCheck();
  navmesh.postProcess();
  return navmesh;
}

void NavmeshParser::buildObjectFileInfoMap() {
	const std::string kObjectFileInfoPath = "navmesh\\object.ifo";
  pk2::PK2Entry objectFileInfoEntry = pk2Reader_.getEntry(kObjectFileInfoPath);
  auto objectFileInfoData = pk2Reader_.getEntryData(objectFileInfoEntry);
  std::string str(objectFileInfoData.begin(), objectFileInfoData.end());
  std::istringstream ss(str);

  std::string line;
  if (!std::getline(ss, line)) {
    throw std::runtime_error("Unable to parse header");
  }
  
  int entryCount;
  ss >> entryCount;
  // Flush the newline character
  ss.ignore();

  while (std::getline(ss, line)) {
    ObjectFileInfo info;
    int index;
    char *ptr = line.data();
    index = strtol(ptr, &ptr, 10);
    info.flag = strtol(ptr, &ptr, 16);
    info.filePath = (ptr+2);
    info.filePath.pop_back();
    objectFileInfoMap_.emplace(index, info);
  }
}

void NavmeshParser::parseMapInfo() {
	const std::string kMapInfoPath = "navmesh\\mapinfo.mfo";
  pk2::PK2Entry mapInfoEntry = pk2Reader_.getEntry(kMapInfoPath);
  auto mapInfoData = pk2Reader_.getEntryData(mapInfoEntry);
  std::string str(mapInfoData.begin(), mapInfoData.end());
  std::istringstream ss(str, std::ios::binary);

  // Advance by 12 bytes. Skipping header
  ss.ignore(12);

  parse(ss, mapInfo_.mapWidth);
  parse(ss, mapInfo_.mapHeight);
  int16_t unk0, unk1, unk2, unk3;
  parse(ss, unk0);
  parse(ss, unk1);
  parse(ss, unk2);
  parse(ss, unk3);
  ss.read(reinterpret_cast<char*>(mapInfo_.regionData.data()), mapInfo_.regionData.size());
}

bool NavmeshParser::regionIsEnabled(uint16_t regionId) const {
  // TODO: Remove, temporary hack to reduce the number of regions we parse on startup
  struct InterestingArea {
    InterestingArea(int x, int y) : regionX(x), regionY(y) {}
    InterestingArea(int x, int y, int w, int h) : regionX(x), regionY(y), width(w), height(h) {}
    int regionX, regionY;
    int width{1};
    int height{1};
  };
  struct Comp {
    bool operator()(const InterestingArea &a, const InterestingArea &b) const {
      if (a.regionX == b.regionX) {
        if (a.regionY == b.regionY) {
          if (a.width == b.width) {
            return a.height < b.height;
          } else {
            return a.width < b.width;
          }
        } else {
          return a.regionY < b.regionY;
        }
      } else {
        return a.regionX < b.regionX;
      }
    }
  };
  static const std::set<InterestingArea, Comp> specialAreas = {
    // {148,92,1,2},  // Tarim Basin
    {106,105,4,4},  // Samarkand
    {133,101,3,3},  // Takla temple & bridges
    {126,102,5,4},  // Takla Yarkan main spawn
    {166,96,4,4},   // Jangan
    {152,102,3,3},  // DW
    {130,87,10,10}, // Hotan
    {122,85,10,10},  // Karakoram
    // {45,84,13,12},  // Alexandria
  };
  const auto [regionX, regionY] = math::position::regionXYFromRegionId(regionId);
  bool matchedOne = specialAreas.empty();
  for (const auto &area : specialAreas) {
    const int maxRegionX = area.regionX + area.width-1;
    const int maxRegionY = area.regionY + area.height-1;
    if (regionX < area.regionX || regionX > maxRegionX) {
      continue;
    }
    if (regionY < area.regionY || regionY > maxRegionY) {
      continue;
    }
    matchedOne = true;
    break;
  }
  if (!matchedOne) {
    return false;
  }
  // TODO: </end hack>
  
  // if (region.IsDungeon)
  //   return false;
  // if (region.X >= this.MapWidth || region.Z >= this.MapHeight)
  //   return false;
  return ((mapInfo_.regionData[regionId >> 3] & (uint8_t)(128 >> regionId % 8))) != 0;
}

void NavmeshParser::parseRegion(uint16_t regionId, navmesh::Navmesh &navmesh) {
  if (!regionIsEnabled(regionId)) {
    throw std::runtime_error("Trying to parse navmesh for disabled region "+std::to_string(regionId));
  }

	const std::string kNavmeshFilePathPrefix = "navmesh\\nv_";
	const std::string kNavmeshFilePathSuffix = ".nvm";
  // Construct a path that is prefix + regionId + suffix
  std::ostringstream filePathStringstream(kNavmeshFilePathPrefix, std::ios::ate);
  filePathStringstream << std::hex << regionId << kNavmeshFilePathSuffix;

  pk2::PK2Entry navmeshEntry = pk2Reader_.getEntry(filePathStringstream.str());
  // Old
  // auto navmeshData = pk2Reader_.getEntryData(navmeshEntry);
  // std::string navmeshDataAsString(navmeshData.begin(), navmeshData.end());
  // std::istringstream navmeshDataStringstream(navmeshDataAsString, std::ios::binary);

  // New
  auto navmeshData = pk2Reader_.getEntryDataChar(navmeshEntry);
  std::istringstream navmeshDataStringstream(std::string(navmeshData.data(), navmeshData.size()), std::ios::binary);

  // Advance by 12 bytes. Skipping header
  navmeshDataStringstream.ignore(12);

  // Whole navmesh data
  navmesh::Region region{regionId};
  
  // Parse object instances
  parseRegionObjectInstances(navmeshDataStringstream, region, navmesh);

  // Parse cells
  parseRegionCellQuads(navmeshDataStringstream, region, navmesh);

  // Parse global edges
  parseRegionGlobalEdges(navmeshDataStringstream, region);
  
  // Parse internal edges
  parseRegionInternalEdges(navmeshDataStringstream, region);

  // Tile map
  parseRegionTileMap(navmeshDataStringstream, region);

  parseRegionHeightMap(navmeshDataStringstream, region);

  // Done with parsing this navmesh file
  // Parse object resources
  parseRegionObjectResources(region, navmesh);

  if (!region.sanityCheck()) {
    throw std::runtime_error("Sanity Check Failure. Some assumptions about the region's navmesh didn't hold.");
  }

  navmesh.addRegion(regionId, std::move(region));
}

void NavmeshParser::parseRegionObjectInstances(std::istringstream &navmeshData, navmesh::Region &region, navmesh::Navmesh &navmesh) {
  uint16_t objectCount;
  parse(navmeshData, objectCount);
  region.objectInstanceIds.reserve(objectCount);

  for (int i=0; i<objectCount; ++i) {
    navmesh::ObjectInstance object;
    parse(navmeshData, object.objectId);
    parse(navmeshData, object.center.x);
    parse(navmeshData, object.center.y);
    parse(navmeshData, object.center.z);
    parse(navmeshData, object.type);
    parse(navmeshData, object.yaw);
    parse(navmeshData, object.localUid);
    parse(navmeshData, object.unk);
    parse(navmeshData, object.isLarge);
    parse(navmeshData, object.isStructure);
    parse(navmeshData, object.regionId);

    // If this object instance belongs in another region, shift the center over for that region
    if (object.regionId != region.id) {
      // The given center is for our region, we should instead save it as a center for the owning region
      const auto [ourRegionX, ourRegionY] = math::position::regionXYFromRegionId(region.id);
      const auto [owningRegionX, owningRegionY] = math::position::regionXYFromRegionId(object.regionId);
      const int dx = (ourRegionX - owningRegionX) * 1920;
      const int dy = (ourRegionY - owningRegionY) * 1920;
      object.center.x += dx;
      object.center.z += dy;
    }

    uint16_t globalEdgeLinkCount;
    parse(navmeshData, globalEdgeLinkCount);
    object.globalEdgeLinks.reserve(globalEdgeLinkCount);
    for (int j=0; j<globalEdgeLinkCount; ++j) {
      navmesh::GlobalEdgeLink edge;
      parse(navmeshData, edge.linkedObjId);
      parse(navmeshData, edge.linkedObjEdgeId);
      parse(navmeshData, edge.edgeId);
      object.globalEdgeLinks.push_back(edge);
    }

    region.objectInstanceIds.push_back(object.globalId());
    navmesh.addObjectInstance(object);
  }
}

void NavmeshParser::parseRegionObjectResources(navmesh::Region &region, navmesh::Navmesh &navmesh) {
  for (const auto objectGlobalId : region.objectInstanceIds) {
    const auto &objectInstance = navmesh.getObjectInstance(objectGlobalId);
    if (!navmesh.haveObjectResource(objectInstance.objectId)) {
      // Don't have this object resource yet
      try {
        navmesh::ObjectResource objectResource = parseObjectResource(objectFileInfoMap_.at(objectInstance.objectId).filePath);
        navmesh.addObjectResource(objectInstance.objectId, objectResource);
      } catch (std::runtime_error &ex) {
        std::cout << "parseObjectResource failed: " << ex.what() << '\n';
      } catch (...) {
        // TOOD: Handle
        std::cout << "Failed to parse object resource at file \"" << objectFileInfoMap_.at(objectInstance.objectId).filePath << '\n';
      }
    }
  }
}

void NavmeshParser::parseRegionCellQuads(std::istringstream &navmeshData, navmesh::Region &region, navmesh::Navmesh &navmesh) {
  uint32_t cellCount, cellExtraCount;
  parse(navmeshData, cellCount);
  parse(navmeshData, cellExtraCount);
  region.cellQuads.reserve(cellCount);

  for (uint32_t i=0; i<cellCount; ++i) {
    navmesh::Cell cell;
    parse(navmeshData, cell.xMin);
    parse(navmeshData, cell.zMin);
    parse(navmeshData, cell.xMax);
    parse(navmeshData, cell.zMax);
    region.cellQuads.emplace_back(cell);
    uint8_t objCount;
    parse(navmeshData, objCount);
    for (int j=0; j<objCount; ++j) {
      uint16_t objIndex;
      parse(navmeshData, objIndex);
    }
  }
}

void NavmeshParser::parseRegionGlobalEdges(std::istringstream &navmeshData, navmesh::Region &region) const {
  uint32_t globalEdgeCount;
  parse(navmeshData, globalEdgeCount);
  region.globalEdges.reserve(globalEdgeCount);
  
  for (uint32_t i=0; i<globalEdgeCount; ++i) {
    navmesh::GlobalEdge edge;
    parse(navmeshData, edge.min.x);
    parse(navmeshData, edge.min.z);
    parse(navmeshData, edge.max.x);
    parse(navmeshData, edge.max.z);
    parse(navmeshData, edge.flag);
    parse(navmeshData, edge.assocDirection0);
    parse(navmeshData, edge.assocDirection1);
    parse(navmeshData, edge.assocCell0);
    parse(navmeshData, edge.assocCell1);
    parse(navmeshData, edge.assocRegion0);
    parse(navmeshData, edge.assocRegion1);
    region.globalEdges.push_back(edge);
  }
}

void NavmeshParser::parseRegionInternalEdges(std::istringstream &navmeshData, navmesh::Region &region) const {
  uint32_t internalEdgeCount;
  parse(navmeshData, internalEdgeCount);
  region.internalEdges.reserve(internalEdgeCount);
  
  for (uint32_t i=0; i<internalEdgeCount; ++i) {
    navmesh::InternalEdge edge;
    parse(navmeshData, edge.min.x);
    parse(navmeshData, edge.min.z);
    parse(navmeshData, edge.max.x);
    parse(navmeshData, edge.max.z);
    parse(navmeshData, edge.flag);
    parse(navmeshData, edge.assocDirection0);
    parse(navmeshData, edge.assocDirection1);
    parse(navmeshData, edge.assocCell0);
    parse(navmeshData, edge.assocCell1);
    region.internalEdges.push_back(edge);

    // Add a reference to this edge in the cell(s) it relates to
    if (edge.assocCell0 != -1) {
      region.cellQuads[edge.assocCell0].edges.emplace_back(i);
    }
    if (edge.assocCell1 != -1) {
      region.cellQuads[edge.assocCell1].edges.emplace_back(i);
    }
  }
}

void NavmeshParser::parseRegionTileMap(std::istringstream &navmeshData, navmesh::Region &region) const {
  int cellId;
  uint16_t flag, textureId;
  for (int row=0; row<96; ++row) {
    for (int col=0; col<96; ++col) {
      parse(navmeshData, cellId);
      parse(navmeshData, flag);
      parse(navmeshData, textureId);
      // Set tile as blocked or not
      region.enabledTiles[row][col] = ((flag & 1) == 0);
    }
  }
}

void NavmeshParser::parseRegionHeightMap(std::istringstream &navmeshData, navmesh::Region &region) const {
  // Parse tile vertex height map
  for (int row=0; row<97; ++row) {
    for (int col=0; col<97; ++col) {
      parse(navmeshData, region.tileVertexHeights[row][col]);
    }
  }

  // Parse surface types
  for (int row=0; row<6; ++row) {
    for (int col=0; col<6; ++col) {
      parse(navmeshData, region.surfaceTypes[row][col]);
    }
  }

  // Parse surface heights
  for (int row=0; row<6; ++row) {
    for (int col=0; col<6; ++col) {
      parse(navmeshData, region.surfaceHeights[row][col]);
    }
  }
}

navmesh::ObjectResource NavmeshParser::parseObjectResource(const std::string &path) {
  // Check the file extension to know which type of file we are parsing
  enum class FileType { kBsr, kCpd, kUnknown };
  const std::string kBsrFileExtension = ".bsr";
  const std::string kCpdFileExtension = ".cpd";
  FileType fileType = FileType::kUnknown;
  if (path.size() >= kBsrFileExtension.size()) {
    if (path.compare(path.size()-kBsrFileExtension.size(), kBsrFileExtension.size(), kBsrFileExtension) == 0) {
      fileType = FileType::kBsr;
    }
  } else {
    throw std::runtime_error("Entire file path \""+path+"\" is shorter than file extension \""+kBsrFileExtension+"\"");
  }
  if (path.size() >= kCpdFileExtension.size()) {
    if (path.compare(path.size()-kCpdFileExtension.size(), kCpdFileExtension.size(), kCpdFileExtension) == 0) {
      fileType = FileType::kCpd;
    }
  } else {
    throw std::runtime_error("Entire file path \""+path+"\" is shorter than file extension \""+kCpdFileExtension+"\"");
  }

  if (fileType == FileType::kCpd) {
    // CPD files are comprised of a BSR and some other resources
    // Call a function to extract the BSR and recursively call this function
    return parseCompoundResource(path);
  }

  // File type must be bsr at this point
  if (fileType != FileType::kBsr) {
    throw std::runtime_error("Unable to determine filetype for file \""+path+"\"");
  }

  // Only parsing this bsr to get to the rootmesh
  pk2::PK2Entry bsrFileEntry = pk2Reader_.getEntry(path);
  auto bsrFileData = pk2Reader_.getEntryData(bsrFileEntry);
  std::string bsrFileDataAsString(bsrFileData.begin(), bsrFileData.end());
  std::istringstream bsrFileDataAsStringstream(bsrFileDataAsString, std::ios::binary);

  // Advance by 12 bytes. Skipping header
  bsrFileDataAsStringstream.ignore(12);
  uint32_t ptrMaterial, ptrMesh, ptrSkeleton, ptrAnimation, ptrMeshGroup, ptrAnimationGroup, ptrSoundEffect, ptrBoundingBox;
  parse(bsrFileDataAsStringstream, ptrMaterial);
  parse(bsrFileDataAsStringstream, ptrMesh);
  parse(bsrFileDataAsStringstream, ptrSkeleton);
  parse(bsrFileDataAsStringstream, ptrAnimation);
  parse(bsrFileDataAsStringstream, ptrMeshGroup);
  parse(bsrFileDataAsStringstream, ptrAnimationGroup);
  parse(bsrFileDataAsStringstream, ptrSoundEffect);
  parse(bsrFileDataAsStringstream, ptrBoundingBox);

  uint32_t unk0, unk1, unk2, unk3, unk4;
  parse(bsrFileDataAsStringstream, unk0);
  parse(bsrFileDataAsStringstream, unk1);
  parse(bsrFileDataAsStringstream, unk2);
  parse(bsrFileDataAsStringstream, unk3);
  parse(bsrFileDataAsStringstream, unk4);

  uint32_t type;
  parse(bsrFileDataAsStringstream, type);

  uint32_t nameLength;
  parse(bsrFileDataAsStringstream, nameLength);

  // Name does not always match the filename
  std::string name;
  name.resize(nameLength);
  bsrFileDataAsStringstream.read(&name[0], nameLength);
  std::cout << "Filename: " << path << std::endl;
  std::cout << "Name: " << name << std::endl << std::endl;
  // // Ignoring the name
  // bsrFileDataAsStringstream.ignore(nameLength);

  std::array<uint8_t, 48> unkBuffer;
  bsrFileDataAsStringstream.read(reinterpret_cast<char*>(unkBuffer.data()), unkBuffer.size());

  // Read the path to the navigation mesh for the object
  uint32_t rootMeshPathLength;
  parse(bsrFileDataAsStringstream, rootMeshPathLength);
  std::string rootMeshPath;
  rootMeshPath.resize(rootMeshPathLength);
  bsrFileDataAsStringstream.read(&rootMeshPath[0], rootMeshPathLength);

  navmesh::ObjectResource obj = parseObjectBms(rootMeshPath);
  obj.name = name;

  // More data, dont care at the moment
  return obj;
}

navmesh::ObjectResource NavmeshParser::parseCompoundResource(const std::string &path) {
  // Only parsing this cpd to get to the rootmesh
  pk2::PK2Entry cpdFileEntry = pk2Reader_.getEntry(path);
  auto cpdFileData = pk2Reader_.getEntryData(cpdFileEntry);
  std::string cpdFileDataAsString(cpdFileData.begin(), cpdFileData.end());
  std::istringstream cpdFileDataAsStringstream(cpdFileDataAsString, std::ios::binary);
  
  // Advance by 12 bytes. Skipping header
  cpdFileDataAsStringstream.ignore(12);

  uint32_t pointerCollisionResource;
  parse(cpdFileDataAsStringstream, pointerCollisionResource);

  cpdFileDataAsStringstream.seekg(pointerCollisionResource, std::ios::beg);

  uint32_t collisionResourcePathLength;
  parse(cpdFileDataAsStringstream, collisionResourcePathLength);
  
  std::string collisionResourcePath;
  collisionResourcePath.resize(collisionResourcePathLength);
  cpdFileDataAsStringstream.read(&collisionResourcePath[0], collisionResourcePathLength);

  return parseObjectResource(collisionResourcePath);
}

navmesh::ObjectResource NavmeshParser::parseObjectBms(const std::string &path) {
  pk2::PK2Entry bmsFileEntry = pk2Reader_.getEntry(path);
  auto bmsFileData = pk2Reader_.getEntryData(bmsFileEntry);
  std::string bmsFileDataAsString(bmsFileData.begin(), bmsFileData.end());
  std::istringstream bmsFileDataAsStringstream(bmsFileDataAsString, std::ios::binary);

  // Advance by 12 bytes. Skipping header
  bmsFileDataAsStringstream.ignore(12);
  uint32_t vertexOffset;
  uint32_t skinOffset;
  uint32_t faceOffset;
  uint32_t clothVertexOffset;
  uint32_t clothEdgeOffset;
  uint32_t boundingBoxOffset;
  uint32_t occlusionPortals;
  uint32_t navMeshOffset;
  uint32_t skinedNavMeshOffset;
  uint32_t unknown9Offset;
  uint32_t unkUInt0;
  uint32_t navFlag; //0 = None, 1 = Edge, 2 = Cell, 4 = Event
  parse(bmsFileDataAsStringstream, vertexOffset);
  parse(bmsFileDataAsStringstream, skinOffset);
  parse(bmsFileDataAsStringstream, faceOffset);
  parse(bmsFileDataAsStringstream, clothVertexOffset);
  parse(bmsFileDataAsStringstream, clothEdgeOffset);
  parse(bmsFileDataAsStringstream, boundingBoxOffset);
  parse(bmsFileDataAsStringstream, occlusionPortals);
  parse(bmsFileDataAsStringstream, navMeshOffset);
  parse(bmsFileDataAsStringstream, skinedNavMeshOffset);
  parse(bmsFileDataAsStringstream, unknown9Offset);
  parse(bmsFileDataAsStringstream, unkUInt0);
  parse(bmsFileDataAsStringstream, navFlag);

  if (navMeshOffset == 0) {
    throw std::runtime_error("Wait, no navmesh?");
  }

  bmsFileDataAsStringstream.seekg(navMeshOffset, std::ios::beg);
  uint32_t vertexCount;
  parse(bmsFileDataAsStringstream, vertexCount);
  navmesh::ObjectResource obj;
  obj.vertices.reserve(vertexCount);

  // NavVertices
  for (uint32_t i=0; i<vertexCount; ++i) {
    float x,y,z;
    uint8_t angleIndex;
    parse(bmsFileDataAsStringstream, x);
    parse(bmsFileDataAsStringstream, y);
    parse(bmsFileDataAsStringstream, z);
    parse(bmsFileDataAsStringstream, angleIndex); // Not sure of the use
    obj.vertices.emplace_back(x,y,z);
  }

  // NavCells
  uint32_t cellCount;
  parse(bmsFileDataAsStringstream, cellCount);
  obj.cells.reserve(cellCount);
  for (uint32_t i=0; i<cellCount; ++i) {
    navmesh::PrimMeshNavCell cell;
    parse(bmsFileDataAsStringstream, cell.vertex0);
    parse(bmsFileDataAsStringstream, cell.vertex1);
    parse(bmsFileDataAsStringstream, cell.vertex2);
    uint16_t flag;
    parse(bmsFileDataAsStringstream, flag);
    if (flag != 0) {
      // If you see a non-zero value, tell Daxter which file
      throw std::runtime_error("Whoa! Nonzero flag");
    }
    if (navFlag & 2) {
      uint8_t eventZoneData;
      parse(bmsFileDataAsStringstream, eventZoneData);
      if (eventZoneData != 0) {
        cell.eventZoneData = eventZoneData;
      }
    }
    obj.cells.emplace_back(std::move(cell));
  }

  // NavOutlineEdges
  uint32_t outlineEdgeCount;
  parse(bmsFileDataAsStringstream, outlineEdgeCount);
  obj.outlineEdges.reserve(outlineEdgeCount);
  for (uint32_t i=0; i<outlineEdgeCount; ++i) {
    navmesh::PrimMeshNavEdge edge;
    parse(bmsFileDataAsStringstream, edge.srcVertex);
    parse(bmsFileDataAsStringstream, edge.destVertex);
    parse(bmsFileDataAsStringstream, edge.srcCell);
    parse(bmsFileDataAsStringstream, edge.destCell);
    parse(bmsFileDataAsStringstream, edge.flag);
    if (navFlag & 1) {
      uint8_t eventZoneData;
      parse(bmsFileDataAsStringstream, eventZoneData);
      if (eventZoneData != 0) {
        edge.eventZoneData = eventZoneData;
      }
    }
    obj.outlineEdges.emplace_back(std::move(edge));
  }

  // NavInlineEdges
  uint32_t inlineEdgeCount;
  parse(bmsFileDataAsStringstream, inlineEdgeCount);
  obj.inlineEdges.reserve(inlineEdgeCount);
  for (uint32_t i=0; i<inlineEdgeCount; ++i) {
    navmesh::PrimMeshNavEdge edge;
    parse(bmsFileDataAsStringstream, edge.srcVertex);
    parse(bmsFileDataAsStringstream, edge.destVertex);
    parse(bmsFileDataAsStringstream, edge.srcCell);
    parse(bmsFileDataAsStringstream, edge.destCell);
    parse(bmsFileDataAsStringstream, edge.flag);
    if (navFlag & 1) {
      uint8_t eventZoneData;
      parse(bmsFileDataAsStringstream, eventZoneData);
      if (eventZoneData != 0) {
        edge.eventZoneData = eventZoneData;
      }
    }
    obj.inlineEdges.emplace_back(std::move(edge));
  }

  // More data, dont care at the moment
  return obj;
}

} // namespace pk2::parsing