set(PROJECT_NAME Combined)

################################################################################
# Source groups
################################################################################
set(no_group_source_files
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
    "../common/pk2/ref/mappingShopGroup.cpp"
    "../common/pk2/ref/mappingShopGroup.hpp"
    "../common/pk2/ref/mappingShopWithTab.cpp"
    "../common/pk2/ref/mappingShopWithTab.hpp"
    "../common/pk2/ref/scrapOfPackageItem.cpp"
    "../common/pk2/ref/scrapOfPackageItem.hpp"
    "../common/pk2/ref/shopGood.cpp"
    "../common/pk2/ref/shopGood.hpp"
    "../common/pk2/ref/shopGroup.cpp"
    "../common/pk2/ref/shopGroup.hpp"
    "../common/pk2/ref/shopTab.cpp"
    "../common/pk2/ref/shopTab.hpp"
    "../common/pk2/ref/skill.cpp"
    "../common/pk2/ref/skill.hpp"
    "../common/pk2/ref/teleport.cpp"
    "../common/pk2/ref/teleport.hpp"
    "../common/shared_io.cpp"
    "../common/shared_io.h"
    "src/bot.cpp"
    "src/bot.hpp"
    "src/broker/eventBroker.cpp"
    "src/broker/eventBroker.hpp"
    "src/broker/packetBroker.cpp"
    "src/broker/packetBroker.hpp"
    "src/broker/timerManager.cpp"
    "src/broker/timerManager.hpp"
    "src/config/characterLoginData.hpp"
    "src/config/configData.cpp"
    "src/config/configData.hpp"
    "src/config/iniReader.cpp"
    "src/config/iniReader.hpp"
    "src/event/event.cpp"
    "src/event/event.hpp"
    "src/loader.cpp"
    "src/loader.hpp"
    "src/math/matrix.cpp"
    "src/math/matrix.hpp"
    "src/math/position.cpp"
    "src/math/position.hpp"
    "src/math/vector.cpp"
    "src/math/vector.hpp"
    "src/module/characterInfoModule.cpp"
    "src/module/characterInfoModule.hpp"
    "src/module/chatEventModule.cpp"
    "src/module/chatEventModule.hpp"
    "src/module/loginModule.cpp"
    "src/module/loginModule.hpp"
    "src/module/movementModule.cpp"
    "src/module/movementModule.hpp"
    "src/module/sharedMemoryWriter.cpp"
    "src/module/sharedMemoryWriter.hpp"
    "src/module/skillUseModule.cpp"
    "src/module/skillUseModule.hpp"
    "src/navmesh/navmesh.cpp"
    "src/navmesh/navmesh.hpp"
    "src/navmesh/triangulation/navmeshTriangulation.cpp"
    "src/navmesh/triangulation/navmeshTriangulation.hpp"
    "src/navmesh/triangulation/singleRegionNavmeshTriangulation.cpp"
    "src/navmesh/triangulation/singleRegionNavmeshTriangulation.hpp"
    "src/navmesh/triangulation/singleRegionNavmeshTriangulationState.cpp"
    "src/navmesh/triangulation/singleRegionNavmeshTriangulationState.hpp"
    "src/packet/building/clientAgentActionCommandRequest.cpp"
    "src/packet/building/clientAgentActionCommandRequest.hpp"
    "src/packet/building/clientAgentActionSelectRequest.cpp"
    "src/packet/building/clientAgentActionSelectRequest.hpp"
    "src/packet/building/clientAgentAuthRequest.cpp"
    "src/packet/building/clientAgentAuthRequest.hpp"
    "src/packet/building/clientAgentCharacterMoveRequest.cpp"
    "src/packet/building/clientAgentCharacterMoveRequest.hpp"
    "src/packet/building/clientAgentCharacterSelectionActionRequest.cpp"
    "src/packet/building/clientAgentCharacterSelectionActionRequest.hpp"
    "src/packet/building/clientAgentCharacterSelectionJoinRequest.cpp"
    "src/packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
    "src/packet/building/clientAgentInventoryItemUseRequest.cpp"
    "src/packet/building/clientAgentInventoryItemUseRequest.hpp"
    "src/packet/building/clientAgentInventoryOperationRequest.cpp"
    "src/packet/building/clientAgentInventoryOperationRequest.hpp"
    "src/packet/building/clientGatewayLoginIbuvAnswer.cpp"
    "src/packet/building/clientGatewayLoginIbuvAnswer.hpp"
    "src/packet/building/clientGatewayLoginRequest.cpp"
    "src/packet/building/clientGatewayLoginRequest.hpp"
    "src/packet/building/serverAgentChatUpdate.cpp"
    "src/packet/building/serverAgentChatUpdate.hpp"
    "src/packet/enums/packetEnums.hpp"
    "src/packet/opcode.cpp"
    "src/packet/opcode.hpp"
    "src/packet/parsing/clientAgentActionCommandRequest.cpp"
    "src/packet/parsing/clientAgentActionCommandRequest.hpp"
    "src/packet/parsing/clientAgentCharacterMoveRequest.cpp"
    "src/packet/parsing/clientAgentCharacterMoveRequest.hpp"
    "src/packet/parsing/clientAgentChatRequest.cpp"
    "src/packet/parsing/clientAgentChatRequest.hpp"
    "src/packet/parsing/commonParsing.cpp"
    "src/packet/parsing/commonParsing.hpp"
    "src/packet/parsing/packetParser.cpp"
    "src/packet/parsing/packetParser.hpp"
    "src/packet/parsing/parsedPacket.cpp"
    "src/packet/parsing/parsedPacket.hpp"
    "src/packet/parsing/serverAgentActionCommandResponse.cpp"
    "src/packet/parsing/serverAgentActionCommandResponse.hpp"
    "src/packet/parsing/serverAgentActionSelectResponse.cpp"
    "src/packet/parsing/serverAgentActionSelectResponse.hpp"
    "src/packet/parsing/serverAgentBuffAdd.cpp"
    "src/packet/parsing/serverAgentBuffAdd.hpp"
    "src/packet/parsing/serverAgentBuffRemove.cpp"
    "src/packet/parsing/serverAgentBuffRemove.hpp"
    "src/packet/parsing/serverAgentCharacterData.cpp"
    "src/packet/parsing/serverAgentCharacterData.hpp"
    "src/packet/parsing/serverAgentChatUpdate.cpp"
    "src/packet/parsing/serverAgentChatUpdate.hpp"
    "src/packet/parsing/serverAgentEntitySyncPosition.cpp"
    "src/packet/parsing/serverAgentEntitySyncPosition.cpp"
    "src/packet/parsing/serverAgentEntityUpdateMovement.cpp"
    "src/packet/parsing/serverAgentEntityUpdateMovement.hpp"
    "src/packet/parsing/serverAgentEntityUpdateMoveSpeed.cpp"
    "src/packet/parsing/serverAgentEntityUpdateMoveSpeed.hpp"
    "src/packet/parsing/serverAgentEntityUpdatePosition.cpp"
    "src/packet/parsing/serverAgentEntityUpdatePosition.hpp"
    "src/packet/parsing/serverAgentEntityUpdateState.cpp"
    "src/packet/parsing/serverAgentEntityUpdateState.hpp"
    "src/packet/parsing/serverAgentSkillBegin.cpp"
    "src/packet/parsing/serverAgentSkillBegin.hpp"
    "src/packet/parsing/serverAgentSkillEnd.cpp"
    "src/packet/parsing/serverAgentSkillEnd.hpp"
    "src/packet/structures/packetInnerStructures.hpp"
    "src/packetLogger.cpp"
    "src/packetLogger.hpp"
    "src/phConnector.cpp"
    "src/pk2/characterData.cpp"
    "src/pk2/characterData.hpp"
    "src/pk2/gameData.cpp"
    "src/pk2/gameData.hpp"
    "src/pk2/itemData.cpp"
    "src/pk2/itemData.hpp"
    "src/pk2/shopData.cpp"
    "src/pk2/shopData.hpp"
    "src/pk2/skillData.cpp"
    "src/pk2/skillData.hpp"
    "src/pk2/teleportData.cpp"
    "src/pk2/teleportData.hpp"
    "src/pk2/parsing/navmeshParser.cpp"
    "src/pk2/parsing/navmeshParser.hpp"
    "src/proxy.cpp"
    "src/proxy.hpp"
    "src/session.cpp"
    "src/session.hpp"
    "src/shared/silkroad_security.cpp"
    "src/shared/silkroad_security.h"
    "src/shared/stream_utility.cpp"
    "src/shared/stream_utility.h"
    "src/silkroadConnection.cpp"
    "src/silkroadConnection.hpp"
    "src/state/entity.cpp"
    "src/state/entity.hpp"
    "src/state/self.cpp"
    "src/state/self.hpp"
    "src/storage/buybackQueue.cpp"
    "src/storage/buybackQueue.hpp"
    "src/storage/item.cpp"
    "src/storage/item.hpp"
    "src/storage/itemList.cpp"
    "src/storage/itemList.hpp"
    "src/storage/storage.cpp"
    "src/storage/storage.hpp"
    "src/ui/userInterface.cpp"
    "src/ui/userInterface.hpp"
)
source_group("" FILES ${no_group_source_files})

set(ALL_FILES
    ${no_group_source_files}
)

################################################################################
# Required packages
################################################################################
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.67.0 REQUIRED COMPONENTS thread regex)
include_directories(${Boost_INCLUDE_DIRS})

include(FindProtobuf)
add_subdirectory(ui-proto)
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Protobuf_INCLUDE_DIRS}
)

find_package(cppzmq CONFIG REQUIRED)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
set(ROOT_NAMESPACE Combined)

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
        "_CONSOLE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
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
        "_CONSOLE"
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
            /DNOMINMAX;
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
            /DNOMINMAX;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /INCREMENTAL
            >
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF;
                /INCREMENTAL:NO
            >
            /DEBUG;
            /SUBSYSTEM:CONSOLE
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /INCREMENTAL
            >
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF;
                /INCREMENTAL:NO
            >
            /DEBUG;
            /SUBSYSTEM:CONSOLE
        )
    endif()
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/src"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>/src"
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    ${Boost_LIBRARIES}
    Pathfinder
    proto
    ${PROTOBUF_LIBRARIES}
    cppzmq
    cppzmq-static
)

# ==============================================================================
# =================================== Tests ====================================
# ==============================================================================

find_package(GTest CONFIG)

if (NOT ${GTest_FOUND})

  # GoogleTest is not installed, pull a copy
  # TODO: Maybe this is a bad idea as it could result in different versions
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/703bd9caab50b139428cea1aaff9974ebee5742e.zip
  )

  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

endif()

add_executable(
  new_test_name_todo_rename
  test/singleRegionNavmeshTriangulationTest.cpp
)

target_link_libraries(
  new_test_name_todo_rename
  Pathfinder
  gtest_main
)

target_include_directories(new_test_name_todo_rename PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/src"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>/src"
)

include(GoogleTest)
gtest_discover_tests(new_test_name_todo_rename
# WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)