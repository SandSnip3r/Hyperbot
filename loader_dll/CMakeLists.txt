set(PROJECT_NAME loader_dll)

################################################################################
# Source groups
################################################################################

set(detours
  "detours/creatwth.cpp"
  "detours/detours.cpp"
  "detours/detours.h"
  "detours/disasm.cpp"
  "detours/image.cpp"
  "detours/modules.cpp"
)
source_group("Common\\detours" FILES ${detours})

set(Source_Files
  "loaderDll.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
  ${detours}
  ${Source_Files}
)

################################################################################
# Required packages
################################################################################
find_package(Boost 1.67.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} SHARED ${ALL_FILES})

target_link_libraries(${PROJECT_NAME}
PRIVATE
  silkroad_lib
)

# add_precompiled_header(${PROJECT_NAME} "stdafx.h" ".")

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
set(ROOT_NAMESPACE ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES
  VS_GLOBAL_KEYWORD "Win32Proj"
)
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
  set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
  )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
  set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
  )
endif()
################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    "$<$<CONFIG:Debug>:"
        "DETOURS_X86;"
        "_CRT_SECURE_NO_WARNINGS;"
        "WIN32;"
        "_DEBUG;"
        "_MBCS"
    ">"
    "$<$<CONFIG:Release>:"
        "NDEBUG;"
        "UNICODE;"
        "_UNICODE"
    ">"
    "LOADERDLL_EXPORTS;"
    "_WINDOWS;"
    "_USRDLL"
  )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    "$<$<CONFIG:Debug>:"
        "DETOURS_X86;"
        "_CRT_SECURE_NO_WARNINGS;"
        "_DEBUG;"
        "_MBCS"
    ">"
    "$<$<CONFIG:Release>:"
        "NDEBUG;"
        "UNICODE;"
        "_UNICODE"
    ">"
    "WIN32;"
    "LOADERDLL_EXPORTS;"
    "_WINDOWS;"
    "_USRDLL"
  )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
  if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_options(${PROJECT_NAME} PRIVATE
      $<$<CONFIG:Debug>:
          /Od;
          /Y-
      >
      $<$<CONFIG:Release>:
          /O2;
          /Oi;
          /Gy
      >
      /permissive-;
      /std:c++17;
      /sdl;
      /W3;
      ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
      ${DEFAULT_CXX_EXCEPTION_HANDLING}
    )
  elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_compile_options(${PROJECT_NAME} PRIVATE
      $<$<CONFIG:Debug>:
          /Od;
          /Y-
      >
      $<$<CONFIG:Release>:
          /O2;
          /Oi;
          /Gy
      >
      /permissive-;
      /std:c++17;
      /sdl;
      /W3;
      ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
      ${DEFAULT_CXX_EXCEPTION_HANDLING}
    )
  endif()
  if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_link_options(${PROJECT_NAME} PRIVATE
      $<$<CONFIG:Release>:
          /OPT:REF;
          /OPT:ICF
      >
      /DEBUG;
      /SUBSYSTEM:WINDOWS;
      /INCREMENTAL:NO
    )
  elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_link_options(${PROJECT_NAME} PRIVATE
      $<$<CONFIG:Release>:
          /OPT:REF;
          /OPT:ICF
      >
      /DEBUG;
      /SUBSYSTEM:WINDOWS;
      /INCREMENTAL:NO
    )
  endif()
endif()

