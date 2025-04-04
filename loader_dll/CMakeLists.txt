project(loader_dll LANGUAGES CXX)

if (NOT ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_SIZEOF_VOID_P EQUAL 4)))
  message(FATAL_ERROR "This project is only supported on Windows x86")
endif()

# Always build the injected DLL as a Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

find_package(Boost CONFIG 1.86.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED
  detours/creatwth.cpp
  detours/detours.cpp
  detours/detours.h
  detours/disasm.cpp
  detours/image.cpp
  detours/modules.cpp
  loaderDll.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  silkroad_lib
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  DETOURS_X86
)