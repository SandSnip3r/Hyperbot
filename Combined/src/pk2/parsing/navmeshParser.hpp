#ifndef PK2_PARSING_NAVMESHPARSER_HPP_
#define PK2_PARSING_NAVMESHPARSER_HPP_

#include "../../navmesh/navmesh.hpp"
#include "../../../../common/pk2/pk2ReaderModern.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>

namespace pk2::parsing {

struct MapInfo {
public:
  uint16_t mapWidth, mapHeight;
  std::array<uint8_t,8192> regionData;
};

struct ObjectFileInfo {
  bool flag;
  std::string filePath;
};

class NavmeshParser {
public:
  NavmeshParser(pk2::Pk2ReaderModern &pk2Reader);
  navmesh::Navmesh parseNavmesh();

private:
  pk2::Pk2ReaderModern &pk2Reader_;
  std::map<int, ObjectFileInfo> objectFileInfoMap_;
  MapInfo mapInfo_;

  void buildObjectFileInfoMap();
  void parseMapInfo();
  bool regionIsEnabled(uint16_t regionId) const;
  void parseRegion(uint16_t regionId, navmesh::Navmesh &navmesh);
  void parseRegionObjectInstances(std::istringstream &navmeshData, navmesh::Region &region, navmesh::Navmesh &navmesh);
  void parseRegionObjectResources(navmesh::Region &region, navmesh::Navmesh &navmesh);
  void parseRegionCellQuads(std::istringstream &navmeshData, navmesh::Region &region, navmesh::Navmesh &navmesh);
  void parseRegionGlobalEdges(std::istringstream &navmeshData, navmesh::Region &region) const;
  void parseRegionInternalEdges(std::istringstream &navmeshData, navmesh::Region &region) const;
  void parseRegionTileMap(std::istringstream &navmeshData, navmesh::Region &region) const;
  void parseRegionHeightMap(std::istringstream &navmeshData, navmesh::Region &region) const;
  
  navmesh::ObjectResource parseObjectResource(const std::string &path);
  navmesh::ObjectResource parseCompoundResource(const std::string &path);
  navmesh::ObjectResource parseObjectBms(const std::string &path);
};

} // namespace pk2::parsing

#endif // PK2_PARSING_NAVMESHPARSER_HPP_