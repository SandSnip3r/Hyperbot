project(silkroad_lib VERSION 0.1 LANGUAGES CXX)

set(PROJECT_SOURCES
  src/blowfish.cc
  src/entity.cc
  src/position.cc
  src/position_math.cc
  src/shared_io.cc
  src/math/matrix4x4.cc
  src/math/vector3.cc
  src/navmesh/navmesh.cc
  src/navmesh/triangulation/navmeshTriangulation.cc
  src/navmesh/triangulation/singleRegionNavmeshTriangulation.cc
  src/navmesh/triangulation/singleRegionNavmeshTriangulationState.cc
  src/pk2/navmeshParser.cc
  src/pk2/pk2Reader.cc
  src/pk2/pk2ReaderModern.cc
)

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
)

add_library(${PROJECT_NAME} STATIC
  ${PROJECT_SOURCES}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  Pathfinder
)

# TODO: Make private
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)

# ==============================================================================
# =================================== Tests ====================================
# ==============================================================================
find_package(GTest REQUIRED)

enable_testing()

set(TEST_NAME ${PROJECT_NAME}_test)

add_executable(
  ${TEST_NAME}
  test/test.cc
)

target_link_libraries(
  ${TEST_NAME}
  ${PROJECT_NAME}
  GTest::Main
)

include(GoogleTest)
gtest_discover_tests(${TEST_NAME}
WORKING_DIRECTORY
${CMAKE_CURRENT_SOURCE_DIR})