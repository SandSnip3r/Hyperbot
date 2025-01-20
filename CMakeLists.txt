cmake_minimum_required(VERSION 3.20)

project(hyperbot_top_level)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(USE_ASAN "Enable AddressSanitizer" OFF)

if(USE_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

find_package(GTest REQUIRED)

add_subdirectory(bot)
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