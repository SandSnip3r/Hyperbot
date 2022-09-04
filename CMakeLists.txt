cmake_minimum_required(VERSION 3.5)

project(SilkroadLib VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBRARY_NAME
  silkroad_lib
)

set(PROJECT_SOURCES
  src/position.cc
  src/position_math.cc
)

add_library(${PROJECT_NAME} STATIC
  ${PROJECT_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
  $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ==============================================================================
# =================================== Tests ====================================
# ==============================================================================

find_package(GTest CONFIG)

if (NOT ${GTest_FOUND})

  # GoogleTest is not installed, pull a copy
  # TODO: Maybe this is a bad idea as it could result in different versions
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/703bd9caab50b139428cea1aaff9974ebee5742e.zip
  )

  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

endif()

enable_testing()

add_executable(
  SilkroadLibTest
  test/test.cc
)

target_link_libraries(
  SilkroadLibTest
  ${PROJECT_NAME}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(SilkroadLibTest
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})