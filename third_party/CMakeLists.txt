set(ABSL_PROPAGATE_CXX_STD ON)

add_subdirectory(abseil-cpp)
add_subdirectory(pathfinder)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_subdirectory(pybind11)
endif()
add_subdirectory(tracy)