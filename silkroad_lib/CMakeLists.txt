project(silkroad_lib VERSION 0.1 LANGUAGES CXX)

set(PROJECT_SOURCES
  src/blowfish.cpp
  src/entity.cpp
  src/file_util.cpp
  src/position.cpp
  src/position_math.cpp
  src/shared_io.cpp
  src/storage.cpp
  src/math/matrix4x4.cpp
  src/math/vector3.cpp
  src/navmesh/navmesh.cpp
  src/navmesh/triangulation/navmeshTriangulation.cpp
  src/navmesh/triangulation/singleRegionNavmeshTriangulation.cpp
  src/navmesh/triangulation/singleRegionNavmeshTriangulationState.cpp
  src/pk2/divisionInfo.cpp
  src/pk2/navmeshParser.cpp
  src/pk2/pk2Reader.cpp
  src/pk2/pk2ReaderModern.cpp
  src/pk2/parsing/parsing.cpp
  src/pk2/parsing/regionInfoParser.cpp
  src/pk2/regionInfo.cpp
  src/pk2/ref/character.cpp
  src/pk2/ref/item.cpp
  src/pk2/ref/level.cpp
  src/pk2/ref/magicOption.cpp
  src/pk2/ref/mappingShopGroup.cpp
  src/pk2/ref/mappingShopWithTab.cpp
  src/pk2/ref/mastery.cpp
  src/pk2/ref/region.cpp
  src/pk2/ref/scrapOfPackageItem.cpp
  src/pk2/ref/shopGood.cpp
  src/pk2/ref/shopGroup.cpp
  src/pk2/ref/shopTab.cpp
  src/pk2/ref/skill.cpp
  src/pk2/ref/teleport.cpp
  src/pk2/ref/text.cpp
  src/pk2/ref/textZoneName.cpp
)

if ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_SIZEOF_VOID_P EQUAL 4))
  list(APPEND PROJECT_SOURCES
    src/edx_labs.cpp
  )
endif()

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
)

add_library(${PROJECT_NAME} STATIC
  ${PROJECT_SOURCES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  Pathfinder
  absl::str_format
)

# Create include/silkroad_lib directory structure at build time
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/silkroad_lib)
file(GLOB_RECURSE HEADER_FILES "src/*.hpp")
foreach(HEADER ${HEADER_FILES})
  file(RELATIVE_PATH REL_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src ${HEADER})
  configure_file(${HEADER} ${CMAKE_CURRENT_BINARY_DIR}/include/silkroad_lib/${REL_HEADER} COPYONLY)
endforeach()

# Configure include paths for this library and clients
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)

# ==============================================================================
# =================================== Tests ====================================
# ==============================================================================

enable_testing()

set(TEST_NAME ${PROJECT_NAME}_test)

add_executable(
  ${TEST_NAME}
  test/test.cpp
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