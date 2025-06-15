project(silkroad_lib VERSION 0.1 LANGUAGES CXX)

set(PROJECT_SOURCES
  src/silkroad_lib/blowfish.cpp
  src/silkroad_lib/entity.cpp
  src/silkroad_lib/file_util.cpp
  src/silkroad_lib/position.cpp
  src/silkroad_lib/position_math.cpp
  src/silkroad_lib/shared_io.cpp
  src/silkroad_lib/storage.cpp
  src/silkroad_lib/math/matrix4x4.cpp
  src/silkroad_lib/math/vector3.cpp
  src/silkroad_lib/navmesh/navmesh.cpp
  src/silkroad_lib/navmesh/triangulation/navmeshTriangulation.cpp
  src/silkroad_lib/navmesh/triangulation/singleRegionNavmeshTriangulation.cpp
  src/silkroad_lib/navmesh/triangulation/singleRegionNavmeshTriangulationState.cpp
  src/silkroad_lib/pk2/divisionInfo.cpp
  src/silkroad_lib/pk2/navmeshParser.cpp
  src/silkroad_lib/pk2/pk2Reader.cpp
  src/silkroad_lib/pk2/pk2ReaderModern.cpp
  src/silkroad_lib/pk2/parsing/helper.cpp
  src/silkroad_lib/pk2/parsing/parsing.cpp
  src/silkroad_lib/pk2/parsing/regionInfoParser.cpp
  src/silkroad_lib/pk2/regionInfo.cpp
  src/silkroad_lib/pk2/ref/character.cpp
  src/silkroad_lib/pk2/ref/item.cpp
  src/silkroad_lib/pk2/ref/level.cpp
  src/silkroad_lib/pk2/ref/magicOption.cpp
  src/silkroad_lib/pk2/ref/mappingShopGroup.cpp
  src/silkroad_lib/pk2/ref/mappingShopWithTab.cpp
  src/silkroad_lib/pk2/ref/mastery.cpp
  src/silkroad_lib/pk2/ref/region.cpp
  src/silkroad_lib/pk2/ref/scrapOfPackageItem.cpp
  src/silkroad_lib/pk2/ref/shopGood.cpp
  src/silkroad_lib/pk2/ref/shopGroup.cpp
  src/silkroad_lib/pk2/ref/shopTab.cpp
  src/silkroad_lib/pk2/ref/skill.cpp
  src/silkroad_lib/pk2/ref/teleport.cpp
  src/silkroad_lib/pk2/ref/text.cpp
  src/silkroad_lib/pk2/ref/textZoneName.cpp
  src/silkroad_lib/pk2/characterData.cpp
  src/silkroad_lib/pk2/gameData.cpp
  src/silkroad_lib/pk2/itemData.cpp
  src/silkroad_lib/pk2/levelData.cpp
  src/silkroad_lib/pk2/magicOptionData.cpp
  src/silkroad_lib/pk2/masteryData.cpp
  src/silkroad_lib/pk2/refRegion.cpp
  src/silkroad_lib/pk2/shopData.cpp
  src/silkroad_lib/pk2/skillData.cpp
  src/silkroad_lib/pk2/teleportData.cpp
  src/silkroad_lib/pk2/textData.cpp
  src/silkroad_lib/pk2/textZoneNameData.cpp
)

if ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_SIZEOF_VOID_P EQUAL 4))
  list(APPEND PROJECT_SOURCES
    src/silkroad_lib/edx_labs.cpp
  )
endif()

add_library(${PROJECT_NAME} STATIC
  ${PROJECT_SOURCES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  Pathfinder
  absl::str_format
  gli
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
  ${CMAKE_CURRENT_SOURCE_DIR}
)