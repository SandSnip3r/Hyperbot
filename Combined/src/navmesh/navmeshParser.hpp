#ifndef NAVMESH_PARSER_HPP_
#define NAVMESH_PARSER_HPP_

#include "../math/vector.hpp"
#include "../../../common/pk2/pk2ReaderModern.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <vector>

struct MapInfo {
public:
  uint16_t mapWidth, mapHeight;
  std::array<uint8_t,8192> regionData;
};

struct GlobalEdgeLink {
  int16_t linkedObjId, linkedObjEdgeId, edgeId;
};

struct MapObjInfo {
  uint32_t objectId;
  Vector center;
  int16_t type;
  float yaw; // Clockwise
  uint16_t localUId;
  int16_t unk;
  uint8_t isLarge, isStructure;
  uint16_t regionId;
  std::vector<GlobalEdgeLink> globalEdgeLinks;
};

struct Cell {
  float xMin, zMin, xMax, zMax;
  std::vector<uint16_t> edges;
};

struct Edge {
  Vector min, max;
  uint8_t flag, assocDirection0, assocDirection1;
  int16_t assocCell0, assocCell1;
};

struct GlobalEdge : public Edge {
  int16_t assocRegion0, assocRegion1;
};

struct InternalEdge : public Edge {};

struct RegionNavmesh {
  std::vector<MapObjInfo> mapObjInfos;
  std::vector<Cell> cellQuads;
  std::vector<GlobalEdge> globalEdges;
  std::vector<InternalEdge> internalEdges;
};

struct ObjectFileInfo {
  bool flag;
  std::string filePath;
};

struct PrimMeshNavCell {
  // PrimMeshNavCell(uint16_t v0, uint16_t v1, uint16_t v2, std::optional<uint8_t> data={}) : vertex0(v0), vertex1(v1), vertex2(v2) {
  //   if (data) {
  //     eventZoneData = *data;
  //   }
  // }
  uint16_t vertex0, vertex1, vertex2;
  std::optional<uint8_t> eventZoneData;
};

struct PrimMeshNavEdge {
  uint16_t srcVertex, destVertex, srcCell, destCell;
  uint8_t flag;
  std::optional<uint8_t> eventZoneData;
};

struct ObjectResource {
  std::vector<Vector> vertices;
  std::vector<PrimMeshNavCell> cells;
  std::vector<PrimMeshNavEdge> outlineEdges, inlineEdges;
};

class NavmeshParser {
public:
  NavmeshParser(pk2::Pk2ReaderModern &pk2Reader);
  bool regionIsEnabled(uint16_t regionId) const;
  RegionNavmesh parseRegionNavmesh(uint16_t regionId);
  const std::map<uint16_t, ObjectResource>& getObjectResourceMap() const;
  const std::map<uint32_t, MapObjInfo>& getObjectInstanceMap() const;
  static constexpr const char* kPathPrefix = "C:/Users/Victor/Documents/Development/PK2 Tools/PK2Tools_0_1/Data/";
  // static constexpr const char* kPathPrefix = "../PK2 Tools/PK2Tools_0_1/Data/";
private:
  pk2::Pk2ReaderModern &pk2Reader_;
  std::map<int, ObjectFileInfo> objectFileInfoMap_;
  MapInfo mapInfo_;
  std::map<uint16_t, ObjectResource> objectResourceMap_;
  std::map<uint32_t, MapObjInfo> objectInstanceMap_;
  void buildObjectFileInfoMap();
  void parseMapInfo();
  void addObjectInstance(const MapObjInfo &object);
  void parseNavmeshMapObjInfos(std::istringstream &navmeshData, std::vector<MapObjInfo> &mapObjInfos);
  void parseNavmeshCellQuads(std::istringstream &navmeshData, RegionNavmesh &navmesh);
  void parseNavmeshGlobalEdges(std::istringstream &navmeshData, RegionNavmesh &navmesh) const;
  void parseNavmeshInternalEdges(std::istringstream &navmeshData, RegionNavmesh &navmesh) const;

  ObjectResource parseObjectResource(const MapObjInfo &object, const std::string &filePath);
  ObjectResource parseCompoundResource(const MapObjInfo &object, const std::string &filePath);
  ObjectResource parseObjectBms(const std::string &filePath);
};

#endif // NAVMESH_PARSER_HPP_