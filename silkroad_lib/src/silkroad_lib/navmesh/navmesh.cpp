#include <silkroad_lib/position_math.hpp>
#include <silkroad_lib/navmesh/navmesh.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stack>
#include <stdexcept>
#include <unordered_set>

namespace sro::navmesh {

namespace {
bool heightInTriangle(const math::Vector3 &point, const math::Vector3 &vertex0, const math::Vector3 &vertex1, const math::Vector3 &vertex2, float &height) {
  const float denom = (vertex1.z-vertex2.z)*(vertex0.x-vertex2.x) + (vertex2.x-vertex1.x)*(vertex0.z-vertex2.z);
  const float w0 = ((vertex1.z-vertex2.z)*(point.x-vertex2.x) + (vertex2.x-vertex1.x)*(point.z-vertex2.z)) / denom;
  const float w1 = ((vertex2.z-vertex0.z)*(point.x-vertex2.x) + (vertex0.x-vertex2.x)*(point.z-vertex2.z)) / denom;
  const float w2 = 1-w0-w1;
  if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) {
    return false;
  } else {
    height = w0*vertex0.y + w1*vertex1.y + w2*vertex2.y;
    return true;
  }
}
} // namespace

float ObjectResource::getHeight(const math::Vector3 &point, const uint32_t areaId) const {
  float height;
  for (int i=0; i<cells.size(); ++i) {
    if (cellAreaIds[i] == areaId) {
      // Does point lie in this cell?
      if (heightInTriangle(point, vertices[cells[i].vertex0], vertices[cells[i].vertex1], vertices[cells[i].vertex2], height)) {
        return height;
      }
    }
  }
  // Point is not on this object
  throw std::runtime_error("This point is not on the object");
}


// ========================================= Navmesh ==========================================

Navmesh::Navmesh(const Navmesh &otherNavmesh): regionMap_(otherNavmesh.regionMap_), objectResourceMap_(otherNavmesh.objectResourceMap_), objectInstanceMap_(otherNavmesh.objectInstanceMap_) {

}

Navmesh& Navmesh::operator=(Navmesh &&otherNavmesh) {
  regionMap_ = std::move(otherNavmesh.regionMap_);
  objectResourceMap_ = std::move(otherNavmesh.objectResourceMap_);
  objectInstanceMap_ = std::move(otherNavmesh.objectInstanceMap_);
  return *this;
}

const std::map<uint16_t, Region>& Navmesh::getRegionMap() const {
  return regionMap_;
}

bool Navmesh::regionIsEnabled(const uint16_t regionId) const {
  return (regionMap_.find(regionId) != regionMap_.end());
}

const Region& Navmesh::getRegion(const uint16_t regionId) const {
  const auto it = regionMap_.find(regionId);
  if (it == regionMap_.end()) {
    // Region not parsed
    throw std::runtime_error("Asking for disabled region");
  }
  return it->second;
}

const ObjectResource& Navmesh::getObjectResource(const uint16_t id) const {
  const auto it = objectResourceMap_.find(id);
  if (it == objectResourceMap_.end()) {
    throw std::runtime_error("Trying to get object resource that does not exist");
  }
  return it->second;
}

const ObjectInstance& Navmesh::getObjectInstance(const uint32_t id) const {
  std::lock_guard<std::mutex> lockGuard(mutex_);
  const auto it = objectInstanceMap_.find(id);
  if (it == objectInstanceMap_.end()) {
    throw std::runtime_error("Trying to get object instance data that does not exist");
  }
  return it->second;
}

math::Matrix4x4 Navmesh::getTransformationFromObjectInstanceToWorld(const uint32_t objectInstanceId, const uint16_t regionId) const {
  const auto &objectInstance = getObjectInstance(objectInstanceId);

  math::Matrix4x4 rotationMatrix;
  rotationMatrix.setRotation(-objectInstance.yaw, {0,1,0});
  math::Matrix4x4 translationMatrix;
  if (objectInstance.regionId == regionId) {
    translationMatrix.setTranslation(objectInstance.center);
  } else {
    // This object isnt for our region, we need to shift it into our region
    const auto [ourRegionX, ourRegionY] = sro::position_math::sectorsFromWorldRegionId(regionId);
    const auto [owningRegionX, owningRegionY] = sro::position_math::sectorsFromWorldRegionId(objectInstance.regionId);
    const int dx = (ourRegionX - owningRegionX) * 1920;
    const int dy = (ourRegionY - owningRegionY) * 1920;
    auto newObjectCenter = objectInstance.center;
    newObjectCenter.x -= dx;
    newObjectCenter.z -= dy;
    translationMatrix.setTranslation(newObjectCenter);
  }
  return translationMatrix*rotationMatrix;

}

ObjectResource Navmesh::getTransformedObjectResourceForRegion(const uint32_t objectInstanceId, const uint16_t regionId) const {
  const auto &objectInstance = getObjectInstance(objectInstanceId);
  auto objectResource = getObjectResource(objectInstance.objectId);
  const auto transformationMatrix = getTransformationFromObjectInstanceToWorld(objectInstanceId, regionId);
  for (auto &vertex : objectResource.vertices) {
    vertex = transformationMatrix*vertex;
  }
  return objectResource;
}

math::Vector3 Navmesh::transformPointIntoObjectFrame(const math::Vector3 &point, const uint16_t regionId, const uint32_t objectInstanceId) const {
  const auto transformationMatrix = getTransformationFromObjectInstanceToWorld(objectInstanceId, regionId);
  return transformationMatrix.inverse()*point;
}

bool Navmesh::haveObjectResource(const uint16_t id) const {
  std::lock_guard<std::mutex> lockGuard(mutex_);
  return (objectResourceMap_.find(id) != objectResourceMap_.end());
}

void Navmesh::addRegion(const uint16_t id, Region &&region) {
  std::lock_guard<std::mutex> lockGuard(mutex_);
  if (regionMap_.find(id) != regionMap_.end()) {
    throw std::runtime_error("Trying to add region which already exists");
  }
  regionMap_.emplace(id, std::move(region));
}

void Navmesh::addObjectInstance(const ObjectInstance &instance) {
  std::lock_guard<std::mutex> lockGuard(mutex_);
  const auto objectGid = instance.globalId();
  auto objectInstanceIt = objectInstanceMap_.find(objectGid);
  if (objectInstanceIt == objectInstanceMap_.end()) {
    // New object, add to map
    objectInstanceMap_.insert(std::make_pair(objectGid, instance));
    return;
  }

  // Object instance already exists, update data
  // From what I've seen, only the global edge links are ever "updated"
  ObjectInstance &existingInstance = objectInstanceIt->second;
  if (instance.globalEdgeLinks.size() != existingInstance.globalEdgeLinks.size()) {
    throw std::runtime_error("Both object instances should have the same number of global edge links");
  }
  for (int i=0; i<instance.globalEdgeLinks.size(); ++i) {
    const auto &oLink = instance.globalEdgeLinks[i];
    auto &eoLink = existingInstance.globalEdgeLinks[i];
    if (oLink.linkedObjId != eoLink.linkedObjId && oLink.linkedObjId != -1) {
      eoLink.linkedObjId = oLink.linkedObjId;
      eoLink.linkedObjGlobalId = oLink.linkedObjGlobalId;
    }
    if (oLink.linkedObjEdgeId != eoLink.linkedObjEdgeId && oLink.linkedObjEdgeId != -1) {
      eoLink.linkedObjEdgeId = oLink.linkedObjEdgeId;
    }
    if (oLink.edgeId != eoLink.edgeId && oLink.edgeId != -1) {
      eoLink.edgeId = oLink.edgeId;
    }
  }
}

void Navmesh::addObjectResource(const uint16_t id, const ObjectResource &resource) {
  std::lock_guard<std::mutex> lockGuard(mutex_);
  if (objectResourceMap_.find(id) != objectResourceMap_.end()) {
    throw std::runtime_error("Adding resource that already exists");
  }
  objectResourceMap_.emplace(id, resource);
}

void Navmesh::sanityCheck() {
  // =================================== Sanity check #1 ====================================
  // For every global region edge, there should be a matching global edge in the neighboring region
  auto edgesAreEqual = [](const auto &edge, const auto srcRegionId, auto potentialEdge, const auto destRegionId) {
    const auto [srcRegionX, srcRegionY] = sro::position_math::sectorsFromWorldRegionId(srcRegionId);
    const auto [destRegionX, destRegionY] = sro::position_math::sectorsFromWorldRegionId(destRegionId);
    if (destRegionX > srcRegionX) {
      // Region to the right
      potentialEdge.min.x = 1920.0;
      potentialEdge.max.x = 1920.0;
    } else if (destRegionX < srcRegionX) {
      // Region to the left
      potentialEdge.min.x = 0.0;
      potentialEdge.max.x = 0.0;
    }
    if (destRegionY > srcRegionY) {
      // Region above
      potentialEdge.min.z = 1920.0;
      potentialEdge.max.z = 1920.0;
    } else if (destRegionY < srcRegionY) {
      // Region below
      potentialEdge.min.z = 0.0;
      potentialEdge.max.z = 0.0;
    }
    bool matches{false};
    bool flagsMatch = (edge.flag == potentialEdge.flag);
    if (edge.min.x == potentialEdge.min.x &&
        edge.min.z == potentialEdge.min.z &&
        edge.max.x == potentialEdge.max.x &&
        edge.max.z == potentialEdge.max.z) {
      // Matching edges
      matches = true;
    } else if (edge.min.x == potentialEdge.max.x &&
               edge.min.z == potentialEdge.max.z &&
               edge.max.x == potentialEdge.min.x &&
               edge.max.z == potentialEdge.min.z) {
      // Matching edges but flipped
      matches = true;
    }
    if (matches) {
      if (!flagsMatch) {
        std::cout << "Found matching edges but flags dont match" << std::endl;
      }
      return true;
    } else {
      return false;
    }
  };
  for (const auto &regionIdRegionPair : regionMap_) {
    for (const auto &edge : regionIdRegionPair.second.globalEdges) {
      const auto otherRegionId = edge.assocRegion1;
      if (regionIsEnabled(otherRegionId)) {
        // This region is also enabled, make sure that there is an exact matching edge
        auto it = regionMap_.find(otherRegionId);
        if (it == regionMap_.end()) {
          throw std::runtime_error("Other region is enabled but not in our map");
        }
        bool foundMatch{false};
        for (const auto &potentialTwinEdge : it->second.globalEdges) {
          if (potentialTwinEdge.assocRegion1 == regionIdRegionPair.first) {
            // This other region's edge references our region, might be a match
            if (edgesAreEqual(edge, regionIdRegionPair.first, potentialTwinEdge, otherRegionId)) {
              foundMatch = true;
              break;
            }
          }
        }
        if (!foundMatch) {
          throw std::runtime_error("Navmesh data does not match our expectations. Global edges do not match in neighboring regions");
        }
      }
    }
  }
  // ========================================================================================

  // =================================== Sanity check #2 ====================================
  // We expect that every linked edge is an unblocked global edge of an object
  //  Note: There are links with bridge edges that exist in the bandit fortress
  //    and links with siege edges that exist in Jangan fortress
  for (const auto &regionIdRegionPair : regionMap_) {
    const auto &region = regionIdRegionPair.second;
    for (const auto objectInstanceId : region.objectInstanceIds) {
      const auto &objectInstance = getObjectInstance(objectInstanceId);
      const auto &objectResource = getObjectResource(objectInstance.objectId);
      for (const auto &edgeLink : objectInstance.globalEdgeLinks) {
        if (edgeLink.edgeId == -1) {
          continue;
        }
        const auto &edgeForOurObject = objectResource.outlineEdges.at(edgeLink.edgeId);
        // if ((edgeForOurObject.flag & static_cast<uint8_t>(EdgeFlag::kBridge)) != 0) {
        //   // TODO: Handle; this happens in bandit fortress
        //   std::cout << "Object " << objectInstanceId << " has an edge link on edge " << edgeLink.edgeId << " that is a bridge edge" << std::endl;
        // }
        // if ((edgeForOurObject.flag & static_cast<uint8_t>(EdgeFlag::kSiege)) != 0) {
        //   // TODO: Handle; this happens in jangan fortress
        //   std::cout << "Object " << objectInstanceId << " has an edge link on edge " << edgeLink.edgeId << " that is a siege edge" << std::endl;
        // }
        if (((edgeForOurObject.flag & static_cast<uint8_t>(EdgeFlag::kBlocked)) != 0)) {
          // Edge flag is blocked!
          throw std::runtime_error("Edge flag is blocked!");
        }
      }
    }
  }
  // ========================================================================================
}

void Navmesh::postProcess() {
  // ================================ Postprocessing step #1 ================================
  //  For each object resource, divide it into areas
  auto setCellAreasForObjectResource = [](auto &objectResource) {
    // Reserve space for the area IDs of every cell
    objectResource.cellAreaIds.resize(objectResource.cells.size());

    std::unordered_set<size_t> visitedCells;

    auto dfsFromCell = [&](const auto cellIndex, const auto areaId) {
      std::stack<size_t> nextCellsStack;
      nextCellsStack.emplace(cellIndex);
      while (!nextCellsStack.empty()) {
        const auto currentCell = nextCellsStack.top();
        nextCellsStack.pop();

        // Mark self as visited
        visitedCells.emplace(currentCell);

        // Set area for this cell
        objectResource.cellAreaIds[currentCell] = areaId;

        // Loop over all edges, if an edge points to another cell, add that cell to be explored in the future
        for (const auto &edge : objectResource.inlineEdges) {
          uint16_t neighborCellIndex;
          if (edge.srcCell == currentCell) {
            neighborCellIndex = edge.destCell;
          } else if (edge.destCell == currentCell) {
            neighborCellIndex = edge.srcCell;
          } else {
            // Edge does not touch our cell
            continue;
          }
          if (neighborCellIndex != -1 && visitedCells.find(neighborCellIndex) == visitedCells.end()) {
            // Add this neighbor to be explored later
            nextCellsStack.emplace(neighborCellIndex);
          }
        }
      }
    };


    uint32_t areaId=0;
    for (size_t i=0; i<objectResource.cells.size(); ++i) {
      if (visitedCells.find(i) == visitedCells.end()) {
        // Unvisited cell
        dfsFromCell(i, areaId);
        ++areaId;
      }
    }

    if (objectResource.cells.size() != visitedCells.size()) {
      throw std::runtime_error("We did not mark areas for every cell in this object");
    }
  };
  for (auto &idResourcePair : objectResourceMap_) {
    setCellAreasForObjectResource(idResourcePair.second);
  }
  // ========================================================================================

  // ================================ Postprocessing step #2 ================================
  // All object instances in a region are not always properly referenced
  // We need to check for any overlap and possibly add to the Region::objectInstanceIds list
  const auto startTime = std::chrono::high_resolution_clock::now();
  auto calculateOverlappingRegions = [](const uint16_t regionId, const float minX, const float minZ, const float maxX, const float maxZ) {
    const auto [regionX, regionY] = sro::position_math::sectorsFromWorldRegionId(regionId);
    const int minRegionX = regionX + static_cast<int>(std::floor(minX/1920.0));
    const int maxRegionX = regionX + static_cast<int>(std::floor(maxX/1920.0));
    const int minRegionY = regionY + static_cast<int>(std::floor(minZ/1920.0));
    const int maxRegionY = regionY + static_cast<int>(std::floor(maxZ/1920.0));
    std::vector<uint16_t> regions;
    regions.reserve((maxRegionX-minRegionX+1) * (maxRegionY-minRegionY+1));
    for (int rx=minRegionX; rx<=maxRegionX; ++rx) {
      for (int ry=minRegionY; ry<=maxRegionY; ++ry) {
        regions.push_back(sro::position_math::worldRegionIdFromSectors(rx, ry));
      }
    }
    return regions;
  };
  auto addObjectInstanceToRegion = [](const auto objectInstanceId, auto &region) {
    if (std::find(region.objectInstanceIds.begin(), region.objectInstanceIds.end(), objectInstanceId) == region.objectInstanceIds.end()) {
      // Object instance was not referenced by this region
      region.objectInstanceIds.reserve(region.objectInstanceIds.size()+1);
      region.objectInstanceIds.push_back(objectInstanceId);
      return true;
    }
    return false;
  };
  for (const auto &objectIdInstancePair : objectInstanceMap_) {
    // Transform the object instance into the region which it belongs
    const auto tranformedObjectResource = getTransformedObjectResourceForRegion(objectIdInstancePair.first, objectIdInstancePair.second.regionId);
    // Create AABB
    float minX = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();
    for (const auto &vertex : tranformedObjectResource.vertices) {
      minX = std::min(minX, vertex.x);
      minX = std::min(minX, vertex.x);
      minZ = std::min(minZ, vertex.z);
      minZ = std::min(minZ, vertex.z);

      maxX = std::max(maxX, vertex.x);
      maxX = std::max(maxX, vertex.x);
      maxZ = std::max(maxZ, vertex.z);
      maxZ = std::max(maxZ, vertex.z);
    }
    // Make sure that this object exists in every region which overlaps with this AABB
    //  This might add the instance in regions which it does not actually overlap, but that's not a problem
    for (const auto regionId : calculateOverlappingRegions(objectIdInstancePair.second.regionId, minX, minZ, maxX, maxZ)) {
      const auto it = regionMap_.find(regionId);
      if (it != regionMap_.end()) {
        // This is an existing region, make sure that this object instance is referenced there
        addObjectInstanceToRegion(objectIdInstancePair.first, it->second);
        // TODO: Is it useful to make sure that the region references objects that are linked to by this object?
        // // Also, make sure that this region references every object that this object links to
        // for (const auto &edgeLink : objectIdInstancePair.second.globalEdgeLinks) {
        //   bool res = addObjectInstanceToRegion(edgeLink.linkedObjGlobalId, it->second);
        //   if (res) {
        //     std::cout << "Object " << objectIdInstancePair.first << " links to object " << edgeLink.linkedObjGlobalId << std::endl;
        //   }
        // }
      }
    }
  }
  const auto endTime = std::chrono::high_resolution_clock::now();
  std::cout << "Adding object instances took " << std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count()/1000.0 << "ms" << std::endl;
  // ========================================================================================
}

// ========================================== Region ==========================================

float Region::getHeightAtPoint(const math::Vector3 &point) const {
  if (point.x < 0.0 || point.x >= 1920.0 ||
      point.z < 0.0 || point.z >= 1920.0) {
    throw std::runtime_error("Trying to get terrain height but point is out of bounds of the region");
  }
  // Use bilinear interpolation to calculate height within tile
  const int tileX = static_cast<int>(point.x/20.0f);
  const int tileZ = static_cast<int>(point.z/20.0f);
  const float xWithinTile = point.x - 20.0f*tileX;
  const float zWithinTile = point.z - 20.0f*tileZ;
  const float xPercent = xWithinTile/20.0f;
  const float heightZ1 = tileVertexHeights[tileZ][tileX] + xPercent * (tileVertexHeights[tileZ][tileX+1] - tileVertexHeights[tileZ][tileX]);
  const float heightZ2 = tileVertexHeights[tileZ+1][tileX] + xPercent * (tileVertexHeights[tileZ+1][tileX+1] - tileVertexHeights[tileZ+1][tileX]);
  const float zPercent = zWithinTile/20.0f;
  float height = heightZ1 + zPercent * (heightZ2 - heightZ1);

  const int surfaceX = static_cast<int>(point.x/320.0f);
  const int surfaceZ = static_cast<int>(point.z/320.0f);
  // Now, check if this is an ice surface
  if ((static_cast<std::underlying_type_t<SurfaceType>>(surfaceTypes[surfaceZ][surfaceX]) & static_cast<std::underlying_type_t<SurfaceType>>(SurfaceType::kIce)) != 0) {
    // Choose the higher of the terrain or the ice surface
    height = std::max(height, surfaceHeights[surfaceZ][surfaceX]);
  }
  return height;
}

bool Region::sanityCheck() const {
  // We assume that every cell's boundaries land on tile boundaries
  for (const auto &cell : cellQuads) {
    for (const float x : { cell.xMin, cell.zMin, cell.xMax, cell.zMax }) {
      if (std::fmod(x,20.0f) != 0) {
        // One of the cell's boundaries doesnt land on a boundary of a tile
        return false;
      }
    }
  }

  // We assume that either every tile inside of a cell is blocked or every tile inside of a cell is not blocked
  for (const auto &cell : cellQuads) {
    bool checkedOneYet{false};
    bool enabled;
    for (int row=static_cast<int>(std::round(cell.zMin/20.0f)), rowEnd=static_cast<int>(std::round(cell.zMax/20.0f))-1; row<=rowEnd; ++row) {
      for (int col=static_cast<int>(std::round(cell.xMin/20.0f)), lastCol=static_cast<int>(std::round(cell.xMax/20.0f))-1; col<=lastCol; ++col) {
        if (checkedOneYet) {
          if (enabled != enabledTiles[row][col]) {
            // A cell contains tiles that are a mixture of blocked/unblocked
            return false;
          }
        } else {
          enabled = enabledTiles[row][col];
          checkedOneYet = true;
        }
      }
    }
  }

  // We assume that every global edge of a region references our region in region0
  for (const auto &edge : globalEdges) {
    if (edge.assocRegion0 != id) {
      throw std::runtime_error("Found a global edge which does not reference our region in region0!");
      return false;
    }
  }
  return true;
}

} // namespace sro::navmesh