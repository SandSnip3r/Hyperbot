cmake_minimum_required(VERSION 3.20)

project(hyperbot_top_level)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)

find_package(GTest CONFIG)

message("CMAKE_SIZEOF_VOID_P: ${CMAKE_SIZEOF_VOID_P}")
message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

# if ((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_SIZEOF_VOID_P EQUAL 4))
  message(STATUS "Configuring Hyperbot")
  add_subdirectory(bot)
# else()
  # message(STATUS "Skipping Hyperbot")
# endif()

set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(abseil-cpp)

add_subdirectory(silkroad_lib)
add_subdirectory(ui_proto)
add_subdirectory(Pathfinder)

option(BUILD_UI "Enable building the UI with Qt" ON)
# cmake -DBUILD_UI=OFF ..

if(BUILD_UI)
  find_package(Qt6 COMPONENTS Core Gui Widgets LinguistTools OpenGLWidgets QUIET)

  if(Qt6_FOUND)
    message(STATUS "Qt6 found, enabling UI build.")
    add_subdirectory(ui)
  else()
    message(WARNING "Qt not found. Skipping UI build.")
  endif()
endif()