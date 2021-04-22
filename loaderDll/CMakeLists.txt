set(PROJECT_NAME loaderDll)

################################################################################
# Source groups
################################################################################
set(Common
    "../common/blowfish.cpp"
    "../common/blowfish.h"
    "../common/Common.cpp"
    "../common/Common.h"
    "../common/pk2/division.cpp"
    "../common/pk2/division.hpp"
    "../common/pk2/divisionInfo.cpp"
    "../common/pk2/divisionInfo.hpp"
    "../common/pk2/parsing/parsing.cpp"
    "../common/pk2/parsing/parsing.hpp"
    "../common/pk2/pk2.h"
    "../common/pk2/pk2Reader.cpp"
    "../common/pk2/pk2Reader.h"
    "../common/pk2/pk2ReaderModern.cpp"
    "../common/pk2/pk2ReaderModern.hpp"
    "../common/pk2/ref/character.cpp"
    "../common/pk2/ref/character.hpp"
    "../common/pk2/ref/item.cpp"
    "../common/pk2/ref/item.hpp"
    "../common/pk2/ref/skill.cpp"
    "../common/pk2/ref/skill.hpp"
    "../common/pk2/ref/teleport.cpp"
    "../common/pk2/ref/teleport.hpp"
    "../common/shared_io.cpp"
    "../common/shared_io.h"
    "../common/shared_types.h"
)
source_group("Common" FILES ${Common})

set(Common__detours
    "../common/detours/creatwth.cpp"
    "../common/detours/detours.cpp"
    "../common/detours/detours.h"
    "../common/detours/disasm.cpp"
    "../common/detours/image.cpp"
    "../common/detours/modules.cpp"
)
source_group("Common\\detours" FILES ${Common__detours})

set(Source_Files
    "loaderDll.cpp"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Common}
    ${Common__detours}
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

# add_precompiled_header(${PROJECT_NAME} "stdafx.h" ".")

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
set(ROOT_NAMESPACE loaderDll)

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

