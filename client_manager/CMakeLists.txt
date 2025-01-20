project(client_manager LANGUAGES CXX)

# This project can only be built for x86 Windows
if (NOT ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_SIZEOF_VOID_P EQUAL 4)))
  message(FATAL_ERROR "${PROJECT_NAME} can only be built for x86 Windows")
endif()

find_package(cppzmq CONFIG REQUIRED)

set(SOURCES
  clientManager.cpp
  clientManager.hpp
  main.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

add_executable(client_manager
  ${SOURCES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  cppzmq
  absl::flags
  absl::flags_parse
  absl::log
  absl::log_initialize
  silkroad_lib
  ui_proto
)
