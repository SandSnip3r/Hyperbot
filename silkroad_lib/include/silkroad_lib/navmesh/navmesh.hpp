#ifndef SRO_NAVMESH_NAVMESH_H_
#define SRO_NAVMESH_NAVMESH_H_

#include "math/matrix4x4.hpp"
#include "math/vector3.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace sro::navmesh {

// ============ Object Instance ============

struct GlobalEdgeLink {
  int16_t linkedObjId, linkedObjEdgeId, edgeId;
  uint32_t linkedObjGlobalId;
};

struct ObjectInstance {
  uint32_t objectId;
  math::Vector3 center;
  int16_t type;
  float yaw; // Clockwise
  uint16_t localUid;
  int16_t unk;
  uint8_t isLarge, isStructure;
  uint16_t regionId;
  std::vector<GlobalEdgeLink> globalEdgeLinks;
  uint32_t globalId() const {
    return ((regionId << 16) | localUid);
  }
};

// =========================================
// ============ Object Resource ============

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
  int16_t srcVertex, destVertex, srcCell, destCell;
  uint8_t flag;
  std::optional<uint8_t> eventZoneData;
};

struct ObjectResource {
  std::string name;
  std::vector<math::Vector3> vertices;
  std::vector<PrimMeshNavCell> cells;
  std::vector<uint32_t> cellAreaIds;
  std::vector<PrimMeshNavEdge> outlineEdges, inlineEdges;
  float getHeight(const math::Vector3 &point, const uint32_t areaId) const;
};

// =========================================
// ================ Region =================

struct Cell {
  float xMin, zMin, xMax, zMax;
  std::vector<uint16_t> edges;
};

enum class EdgeFlag : uint8_t {
  kNone = 0,
  kBlockDst2Src = 1,
  kBlockSrc2Dst = 2,
  kBlocked = kBlockDst2Src | kBlockSrc2Dst,
  kInternal = 4,
  kGlobal = 8,
  kBridge = 16,
  kEntrance = 32,  // Dungeon
  kBit6 = 64,
  kSiege = 128,    // Fortress War (projectile passthrough)
};

enum class EdgeDirection : int8_t {
  kInvalid = -1,
  kNorth = 0,
  kEast = 1,
  kSouth = 2,
  kWest = 3,
};

struct Edge {
  // TODO: Use 2d Vector here instead
  math::Vector3 min, max;
  EdgeFlag flag;
  EdgeDirection assocDirection0, assocDirection1;
  int16_t assocCell0, assocCell1;
};

struct GlobalEdge : public Edge {
  int16_t assocRegion0, assocRegion1;
};

struct InternalEdge : public Edge {};

enum class SurfaceType : uint8_t {
  kNone = 0,
  kWater = 1,
  kIce = 2
};

struct Region {
  explicit Region(uint16_t idParam) : id(idParam) {}
  uint16_t id;
  std::vector<uint32_t> objectInstanceIds;
  std::vector<Cell> cellQuads;
  std::vector<GlobalEdge> globalEdges;
  std::vector<InternalEdge> internalEdges;
  std::array<std::array<bool,96>,96> enabledTiles;
  std::array<std::array<float,97>,97> tileVertexHeights;
  std::array<std::array<SurfaceType,6>,6> surfaceTypes;
  std::array<std::array<float,6>,6> surfaceHeights;
  float getHeightAtPoint(const math::Vector3 &point) const;
  bool sanityCheck() const;
};

// =========================================

class Navmesh {
public:
  Navmesh() = default;
  Navmesh(const Navmesh &otherNavmesh);
  Navmesh& operator=(Navmesh &&otherNavmesh);

  const std::map<uint16_t, Region>& getRegionMap() const;
  bool regionIsEnabled(const uint16_t regionId) const;
  const Region& getRegion(const uint16_t regionId) const;
  const ObjectResource& getObjectResource(const uint16_t id) const;
  const ObjectInstance& getObjectInstance(const uint32_t id) const;
  math::Matrix4x4 getTransformationFromObjectInstanceToWorld(const uint32_t objectInstanceId, const uint16_t regionId) const;
  ObjectResource getTransformedObjectResourceForRegion(const uint32_t objectInstanceId, const uint16_t regionId) const;
  math::Vector3 transformPointIntoObjectFrame(const math::Vector3 &point, const uint16_t regionId, const uint32_t objectInstanceId) const;

  bool haveObjectResource(const uint16_t id) const;

  void addRegion(const uint16_t id, Region &&region);
  void addObjectInstance(const ObjectInstance &instance);
  void addObjectResource(const uint16_t id, const ObjectResource &resource);

  void sanityCheck();
  void postProcess();

private:
  mutable std::mutex mutex_;
  std::map<uint16_t, Region> regionMap_;
  std::map<uint16_t, ObjectResource> objectResourceMap_;
  std::map<uint32_t, ObjectInstance> objectInstanceMap_;
};

} // namespace sro::navmesh

#endif // SRO_NAVMESH_NAVMESH_H_