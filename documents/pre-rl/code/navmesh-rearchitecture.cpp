

// class SingleRegionNavmeshTriangulation : public pathfinder::navmesh::TriangleLibNavmesh {
// public:
//   using State = SingleRegionNavmeshTriangulationState<IndexType>;

//   SingleRegionNavmeshTriangulation(const navmesh::Navmesh &navmesh,
//                                    const navmesh::Region &region,
//                                    const triangle::triangleio &triangleData,
//                                    const triangle::triangleio &triangleVoronoiData,
//                                    std::vector<std::vector<ConstraintData>> &&constraintData,
//                                    const std::vector<ObjectLink> &globalObjectLinks);

#include <iostream>

using namespace std;

namespace silkroad::navmesh {

class Navmesh;
class Region;

class NavmeshTriangulation {
public:
  NavmeshTriangulation(const Navmesh &navmesh) {
    // Triangulate each region, 1-by-1
    for (auto region : navmesh.regions) {
      triangulateRegion(navmesh, region);
    }

    // Do any extra work that needs to be done post-triangulation
    postProcess();
  }
private:
  void triangulateRegion(const Navmesh &navmesh, const Region &region) {
    /************************************************************************************
    ** Step #1
    ** Convert all of the data from the navmesh into an intermediate
    ** representation that we can transform directly into a triangle::triangleio object
    ************************************************************************************/
    // Calculate and add our own global edges from the region
    // Add internal edges of the region
    // Add internal and external edges of all objects

    /************************************************************************************
    ** Step #2
    ** Populate the triangle::triangleio object and triangulate
    ************************************************************************************/
    // Copy vertices, edges, and constraint markers

    /************************************************************************************
    ** Step #3
    ** Construct a SingleRegionNavmeshTriangulation from the triangle::triangleio data
    ************************************************************************************/

    /************************************************************************************
    ** Step #4
    ** Post-process on the SingleRegionNavmeshTriangulation
    ************************************************************************************/
  }
};

} // namespace silkroad::navmesh

int main() {
  /*******************************
  ** Typical workflow:
  *******************************/
  // Misc constants
  const string kSilkroadPath_{"..."};
  const auto kDataPath = kSilkroadPath_ / "Data.pk2";
  const auto kAgentRadius{3.14};
  // Open the Data.pk2 file
  silkroad::pk2::Pk2ReaderModern pk2Reader{kDataPath};
  // Parse all of the navmesh data into an intermediate format
  silkroad::pk2::parsing::NavmeshParser navmeshParser{pk2Reader};
  silkroad::navmesh::Navmesh navmesh{navmeshParser.parseNavmesh()};
  // Triangulate the navmesh
  silkroad::navmesh::NavmeshTriangulation navmeshTriangulation{navmesh};
  // Construct pathfinder with reference to the triangulation
  pathfinder::Pathfinder<navmesh::triangulation::NavmeshTriangulation> pathfinder(navmeshTriangulation, kAgentRadius);
  // Find the shortest path
  auto result = pathfinder.findShortestPath(start, goal);

  /*******************************
  ** Unit testing workflow:
  *******************************/
  SingleRegionNavmeshTriangulation singleRegionTriangulation;
  // TODO: Somehow construct this SingleRegionNavmeshTriangulation without needing a triangle::triangleio
  return 0;
}