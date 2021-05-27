#include "navmeshParser.hpp"

#include <array>
#include <exception>
#include <fstream>
#include <string>
#include <sstream>

#include <iostream>

// ========================================================================
// =========================Basic helper functions=========================
// ========================================================================

template<typename T>
void parse(std::istringstream &inFile, T &data) {
  inFile.read(reinterpret_cast<char*>(&data), sizeof(T));
}

void rectifyPath(std::string &path) {
  // Prepend path
  path = NavmeshParser::kPathPrefix + path;
  // Remove backslashes
  for (int i=0; i<path.size(); ++i) {
    if (path[i] == '\\') {
      path[i] = '/';
    }
  }
}

// ========================================================================
// =========================NavmeshParser functions========================
// ========================================================================

NavmeshParser::NavmeshParser(pk2::Pk2ReaderModern &pk2Reader) : pk2Reader_(pk2Reader) {
  buildObjectFileInfoMap();
  parseMapInfo();
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
  // if (region.IsDungeon)
  //   return false;
  // if (region.X >= this.MapWidth || region.Z >= this.MapHeight)
  //   return false;
  return ((mapInfo_.regionData[regionId >> 3] & (uint8_t)(128 >> regionId % 8))) != 0;
}

RegionNavmesh NavmeshParser::parseRegionNavmesh(uint16_t regionId) {
  if (!regionIsEnabled(regionId)) {
    throw std::runtime_error("Trying to parse navmesh for disabled region "+std::to_string(regionId));
  }

  // TODO: Probably dont need these
  int16_t x = regionId & 0xFF;
  int16_t y = (regionId >> 8) & 0xFF;

	const std::string kNavmeshFilePathPrefix = "navmesh\\nv_";
	const std::string kNavmeshFilePathSuffix = ".nvm";
  // Construct a path that is prefix + regionId + suffix
  std::ostringstream filePathStringstream(kNavmeshFilePathPrefix, std::ios::ate);
  filePathStringstream << std::hex << regionId << kNavmeshFilePathSuffix;

  pk2::PK2Entry navmeshEntry = pk2Reader_.getEntry(filePathStringstream.str());
  auto navmeshData = pk2Reader_.getEntryData(navmeshEntry);
  std::string navmeshDataAsString(navmeshData.begin(), navmeshData.end());
  std::istringstream navmeshDatStringstream(navmeshDataAsString, std::ios::binary);

  // Advance by 12 bytes. Skipping header
  navmeshDatStringstream.ignore(12);

  // Whole navmesh data
  RegionNavmesh navmesh;
  
  // Parse object instances
  parseNavmeshMapObjInfos(navmeshDatStringstream, navmesh.mapObjInfos);

  // Parse cells
  parseNavmeshCellQuads(navmeshDatStringstream, navmesh);

  // Parse global edges
  parseNavmeshGlobalEdges(navmeshDatStringstream, navmesh);
  
  // Parse internal edges
  parseNavmeshInternalEdges(navmeshDatStringstream, navmesh);

  int cellId;
  uint16_t flag, textureId;
  for (int i=0; i<96; ++i) {
    for (int j=0; j<96; ++j) {
      parse(navmeshDatStringstream, cellId);
      parse(navmeshDatStringstream, flag);
      parse(navmeshDatStringstream, textureId);
    }
  }

  float height;
  for (int i=0; i<97; ++i) {
    for (int j=0; j<97; ++j) {
      parse(navmeshDatStringstream, height);
    }
  }

  uint8_t surfaceType; //0 = None, 1 = Water, 2 = Ice
  for (int i=0; i<6; ++i) {
    for (int j=0; j<6; ++j) {
      parse(navmeshDatStringstream, surfaceType);
    }
  }

  float surfaceHeight;
  for (int i=0; i<6; ++i) {
    for (int j=0; j<6; ++j) {
      parse(navmeshDatStringstream, surfaceHeight);
    }
  }

  return navmesh;
}

const std::map<uint16_t, ObjectResource>& NavmeshParser::getObjectResourceMap() const {
  return objectResourceMap_;
}

const std::map<uint32_t, MapObjInfo>& NavmeshParser::getObjectInstanceMap() const {
  return objectInstanceMap_;
}

void NavmeshParser::addObjectInstance(const MapObjInfo &object) {
  uint32_t objectGid = (object.regionId << 16) | object.localUId;
  auto objectInstanceIt = objectInstanceMap_.find(objectGid);
  if (objectInstanceIt == objectInstanceMap_.end()) {
    // New object, add to map
    objectInstanceMap_.insert(std::make_pair(objectGid, object));
    return;
  }

  // Object instance already exists, update data
  // From what I've seen, only the global edge links are ever "updated"
  MapObjInfo &existingObject = objectInstanceIt->second;
  for (int i=0; i<object.globalEdgeLinks.size(); ++i) {
    const auto &oLink = object.globalEdgeLinks[i];
    auto &eoLink = existingObject.globalEdgeLinks[i];
    if (oLink.linkedObjId != eoLink.linkedObjId && oLink.linkedObjId != -1) { 
      eoLink.linkedObjId = oLink.linkedObjId;
    }
    if (oLink.linkedObjEdgeId != eoLink.linkedObjEdgeId && oLink.linkedObjEdgeId != -1) { 
      eoLink.linkedObjEdgeId = oLink.linkedObjEdgeId;
    }
    if (oLink.edgeId != eoLink.edgeId && oLink.edgeId != -1) { 
      eoLink.edgeId = oLink.edgeId;
    }
  }
}

void NavmeshParser::parseNavmeshMapObjInfos(std::istringstream &navmeshData, std::vector<MapObjInfo> &mapObjInfos) {
  uint16_t objectCount;
  parse(navmeshData, objectCount);
  mapObjInfos.reserve(objectCount);

  for (int i=0; i<objectCount; ++i) {
    MapObjInfo object;
    parse(navmeshData, object.objectId);
    parse(navmeshData, object.center.x);
    parse(navmeshData, object.center.y);
    parse(navmeshData, object.center.z);
    parse(navmeshData, object.type);
    parse(navmeshData, object.yaw);
    parse(navmeshData, object.localUId);
    parse(navmeshData, object.unk);
    parse(navmeshData, object.isLarge);
    parse(navmeshData, object.isStructure);
    parse(navmeshData, object.regionId);
    
    uint16_t globalEdgeLinkCount;
    parse(navmeshData, globalEdgeLinkCount);
    object.globalEdgeLinks.reserve(globalEdgeLinkCount);
    for (int j=0; j<globalEdgeLinkCount; ++j) {
      GlobalEdgeLink edge;
      parse(navmeshData, edge.linkedObjId);
      parse(navmeshData, edge.linkedObjEdgeId);
      parse(navmeshData, edge.edgeId);
      object.globalEdgeLinks.push_back(edge);
    }
    
    mapObjInfos.push_back(object);
    addObjectInstance(object);
  }
}

void NavmeshParser::parseNavmeshCellQuads(std::istringstream &navmeshData, RegionNavmesh &navmesh) {
  uint32_t cellCount, cellExtraCount;
  parse(navmeshData, cellCount);
  parse(navmeshData, cellExtraCount);
  navmesh.cellQuads.reserve(cellCount);

  int totalReferences=0;
  for (uint32_t i=0; i<cellCount; ++i) {
    Cell cell;
    parse(navmeshData, cell.xMin);
    parse(navmeshData, cell.zMin);
    parse(navmeshData, cell.xMax);
    parse(navmeshData, cell.zMax);
    navmesh.cellQuads.emplace_back(cell);
    uint8_t objCount;
    parse(navmeshData, objCount);
    for (int j=0; j<objCount; ++j) {
      uint16_t objIndex;
      parse(navmeshData, objIndex);

      // Since we're here and know this object is required, we'll parse this ObjectResource now
      //  This could also be done later in batch after all cells are parsed
      const MapObjInfo &objectInstance = navmesh.mapObjInfos.at(objIndex);
      if (objectResourceMap_.find(objectInstance.objectId) == objectResourceMap_.end()) {
        try {
          ObjectResource objectResource = parseObjectResource(objectInstance, objectFileInfoMap_.at(objectInstance.objectId).filePath);
          objectResourceMap_.emplace(objectInstance.objectId, objectResource);
        } catch (std::runtime_error &ex) {
          std::cout << ex.what() << '\n';
          //
        } catch (...) {
          std::cout << "Failed to parse\n";
          // TOOD: Handle
          // cout << "Failed to parse object resource at file \"" << objectFileInfoMap_.at(objectInstance.objectId).filePath << '\n';
        }
      }
    }
  }
}

void NavmeshParser::parseNavmeshGlobalEdges(std::istringstream &navmeshData, RegionNavmesh &navmesh) const {
  uint32_t globalEdgeCount;
  parse(navmeshData, globalEdgeCount);
  navmesh.globalEdges.reserve(globalEdgeCount);
  
  for (uint32_t i=0; i<globalEdgeCount; ++i) {
    GlobalEdge edge;
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
    navmesh.globalEdges.push_back(edge);
  }
}

void NavmeshParser::parseNavmeshInternalEdges(std::istringstream &navmeshData, RegionNavmesh &navmesh) const {
  uint32_t internalEdgeCount;
  parse(navmeshData, internalEdgeCount);
  navmesh.internalEdges.reserve(internalEdgeCount);
  
  for (uint32_t i=0; i<internalEdgeCount; ++i) {
    InternalEdge edge;
    parse(navmeshData, edge.min.x);
    parse(navmeshData, edge.min.z);
    parse(navmeshData, edge.max.x);
    parse(navmeshData, edge.max.z);
    parse(navmeshData, edge.flag);
    parse(navmeshData, edge.assocDirection0);
    parse(navmeshData, edge.assocDirection1);
    parse(navmeshData, edge.assocCell0);
    parse(navmeshData, edge.assocCell1);
    navmesh.internalEdges.push_back(edge);

    // Add a reference to this edge in the cell(s) it relates to
    if (edge.assocCell0 != -1) {
      navmesh.cellQuads[edge.assocCell0].edges.emplace_back(i);
    }
    if (edge.assocCell1 != -1) {
      navmesh.cellQuads[edge.assocCell1].edges.emplace_back(i);
    }
  }
}

ObjectResource NavmeshParser::parseObjectBms(const std::string &filePath) {
  pk2::PK2Entry bmsFileEntry = pk2Reader_.getEntry(filePath);
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
  ObjectResource obj;
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
    PrimMeshNavCell cell;
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
    PrimMeshNavEdge edge;
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
    PrimMeshNavEdge edge;
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

ObjectResource NavmeshParser::parseCompoundResource(const MapObjInfo &object, const std::string &filePath) {
  // Only parsing this cpd to get to the rootmesh
  pk2::PK2Entry cpdFileEntry = pk2Reader_.getEntry(filePath);
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
  std::cout << "Expecting to find the collision resource at the path \"" << collisionResourcePath << "\"\n";

  return parseObjectResource(object, collisionResourcePath);
}

ObjectResource NavmeshParser::parseObjectResource(const MapObjInfo &object, const std::string &filePath) {
  // Check the file extension to know which type of file we are parsing
  enum class FileType { kBsr, kCpd, kUnknown };
  const std::string kBsrFileExtension = ".bsr";
  const std::string kCpdFileExtension = ".cpd";
  FileType fileType = FileType::kUnknown;
  if (filePath.size() >= kBsrFileExtension.size()) {
    if (filePath.compare(filePath.size()-kBsrFileExtension.size(), kBsrFileExtension.size(), kBsrFileExtension) == 0) {
      fileType = FileType::kBsr;
    }
  } else {
    throw std::runtime_error("Entire file path \""+filePath+"\" is shorter than file extension \""+kBsrFileExtension+"\"");
  }
  if (filePath.size() >= kCpdFileExtension.size()) {
    if (filePath.compare(filePath.size()-kCpdFileExtension.size(), kCpdFileExtension.size(), kCpdFileExtension) == 0) {
      fileType = FileType::kCpd;
    }
  } else {
    throw std::runtime_error("Entire file path \""+filePath+"\" is shorter than file extension \""+kCpdFileExtension+"\"");
  }

  if (fileType == FileType::kCpd) {
    // CPD files are comprised of a BSR and some other resources
    // Call a function to extract the BSR and recursively call this function
    return parseCompoundResource(object, filePath);
  }

  // File type must be bsr at this point
  if (fileType != FileType::kBsr) {
    throw std::runtime_error("Unable to determine filetype for file \""+filePath+"\"");
  }

  // Only parsing this bsr to get to the rootmesh
  pk2::PK2Entry bsrFileEntry = pk2Reader_.getEntry(filePath);
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

  ObjectResource obj = parseObjectBms(rootMeshPath);
  obj.name = name;

  // More data, dont care at the moment
  return obj;
}