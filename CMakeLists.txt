cmake_minimum_required(VERSION 3.20)

project(HyperbotTopLevel)

if ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_GENERATOR_PLATFORM STREQUAL "Win32"))
  add_subdirectory(bot)
endif()

add_subdirectory(SilkroadLib)
add_subdirectory(ui-proto)

if ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (NOT CMAKE_GENERATOR_PLATFORM STREQUAL "x64"))
  message("ADDING UI abc123abc123abc123abc123abc123abc123abc123")

  add_subdirectory(ui)
else()
  message("abc123abc123abc123abc123abc123abc123abc123")
  message("${CMAKE_SYSTEM_NAME}")
  message("${CMAKE_GENERATOR_PLATFORM}")
endif()
