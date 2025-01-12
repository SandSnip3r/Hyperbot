cmake_minimum_required(VERSION 3.20)

project(HyperbotTopLevel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_GENERATOR_PLATFORM STREQUAL "Win32"))
  add_subdirectory(bot)
endif()

set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(abseil-cpp)

add_subdirectory(SilkroadLib)
add_subdirectory(ui-proto)
add_subdirectory(Pathfinder)

if ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (NOT CMAKE_GENERATOR_PLATFORM STREQUAL "Win32"))
  message("ADDING UI abc123abc123abc123abc123abc123abc123abc123")

  add_subdirectory(ui)
else()
  message("abc123abc123abc123abc123abc123abc123abc123")
  message("${CMAKE_SYSTEM_NAME}")
  message("${CMAKE_GENERATOR_PLATFORM}")
endif()
