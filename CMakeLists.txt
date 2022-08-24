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