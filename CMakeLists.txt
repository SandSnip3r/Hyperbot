cmake_minimum_required(VERSION 3.20)

project(hyperbot_top_level)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- make CMake prefer xxx-config.cmake over Findxxx.cmake ---
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# --- make sure the vcpkg copy is found in CONFIG mode ---
# (vcpkg toolchain already adds its share/ path to CMAKE_PREFIX_PATH,
#  so this "find_package" will succeed.)
find_package(Protobuf CONFIG REQUIRED)

option(USE_ASAN "Enable AddressSanitizer" OFF)
option(TRACY_ENABLE "Enable Tracy Profiler" OFF)

if(USE_ASAN)
  # AddressSanitizer does not play well with CUDA! If you must use it, set JAX_PLATFORMS=cpu.
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

find_package(absl CONFIG REQUIRED)
find_package(GTest REQUIRED)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  include_directories(${Python3_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/include)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_subdirectory(bot)
endif()
add_subdirectory(silkroad_lib)
add_subdirectory(third_party)
add_subdirectory(ui_proto)

if ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_SIZEOF_VOID_P EQUAL 4))
  add_subdirectory(client_manager)
  add_subdirectory(loader_dll)
endif()

option(BUILD_UI "Enable building the UI with Qt" ON)

if(BUILD_UI)
  find_package(Qt6 COMPONENTS Core Gui Widgets LinguistTools OpenGLWidgets QUIET)

  if(Qt6_FOUND)
    message(STATUS "Qt6 found, enabling UI build.")
    add_subdirectory(ui)
    add_subdirectory(rl_ui)
  else()
    message(WARNING "Qt not found. Skipping UI build.")
  endif()
endif()