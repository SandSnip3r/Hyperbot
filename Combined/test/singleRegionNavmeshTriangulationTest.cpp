#include "navmesh/triangulation/singleRegionNavmeshTriangulation.hpp"

#include <gtest/gtest.h>

class StartTouchingVertexTest : public ::testing::Test {
protected:
  StartTouchingVertexTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(StartTouchingVertexTest, StartTopGoingLeft) {
}

TEST(Name, name2) {
  // std::vector<SingleRegionNavmeshTriangulation::State> SingleRegionNavmeshTriangulation::getSuccessors(const State &currentState, const std::optional<State> goalState, const double agentRadius) const {
  // ============================================================================================================================================
  using TestType = navmesh::triangulation::SingleRegionNavmeshTriangulation;
  // TestType t;
  std::cout << "We're in the test now!!" << std::endl;
}