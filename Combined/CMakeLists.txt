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
    "../common/pk2/ref/level.cpp"
    "../common/pk2/ref/level.hpp"
    "../common/pk2/ref/magicOption.cpp"
    "../common/pk2/ref/magicOption.hpp"
    "../common/pk2/ref/mappingShopGroup.cpp"
    "../common/pk2/ref/mappingShopGroup.hpp"
    "../common/pk2/ref/mappingShopWithTab.cpp"
    "../common/pk2/ref/mappingShopWithTab.hpp"
    "../common/pk2/ref/region.cpp"
    "../common/pk2/ref/region.hpp"
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
    "../common/pk2/ref/textItemOrSkill.cpp"
    "../common/pk2/ref/textItemOrSkill.hpp"
    "../common/pk2/ref/textZoneName.cpp"
    "../common/pk2/ref/textZoneName.hpp"
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
    "src/config/config.cpp"
    "src/config/config.hpp"
    "src/config/configData.cpp"
    "src/config/configData.hpp"
    "src/config/iniReader.cpp"
    "src/config/iniReader.hpp"
    "src/entity/entity.cpp"
    "src/entity/entity.hpp"
    "src/entity/geometry.cpp"
    "src/entity/geometry.hpp"
    "src/event/event.cpp"
    "src/event/event.hpp"
    "src/helpers.cpp"
    "src/helpers.hpp"
    "src/loader.cpp"
    "src/loader.hpp"
    "src/logging.cpp"
    "src/logging.hpp"
    "src/main.cpp"
    "src/math/pointTranslator.cpp"
    "src/math/pointTranslator.hpp"
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
    "src/packet/building/clientAgentActionDeselectRequest.cpp"
    "src/packet/building/clientAgentActionDeselectRequest.hpp"
    "src/packet/building/clientAgentActionSelectRequest.cpp"
    "src/packet/building/clientAgentActionSelectRequest.hpp"
    "src/packet/building/clientAgentActionTalkRequest.cpp"
    "src/packet/building/clientAgentActionTalkRequest.hpp"
    "src/packet/building/clientAgentAuthRequest.cpp"
    "src/packet/building/clientAgentAuthRequest.hpp"
    "src/packet/building/clientAgentCharacterMoveRequest.cpp"
    "src/packet/building/clientAgentCharacterMoveRequest.hpp"
    "src/packet/building/clientAgentCharacterSelectionActionRequest.cpp"
    "src/packet/building/clientAgentCharacterSelectionActionRequest.hpp"
    "src/packet/building/clientAgentCharacterSelectionJoinRequest.cpp"
    "src/packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
    "src/packet/building/clientAgentCharacterUpdateBodyStateRequest.cpp"
    "src/packet/building/clientAgentCharacterUpdateBodyStateRequest.hpp"
    "src/packet/building/clientAgentCosCommandRequest.cpp"
    "src/packet/building/clientAgentCosCommandRequest.hpp"
    "src/packet/building/clientAgentFreePvpUpdateRequest.cpp"
    "src/packet/building/clientAgentFreePvpUpdateRequest.hpp"
    "src/packet/building/clientAgentInventoryItemUseRequest.cpp"
    "src/packet/building/clientAgentInventoryItemUseRequest.hpp"
    "src/packet/building/clientAgentInventoryOperationRequest.cpp"
    "src/packet/building/clientAgentInventoryOperationRequest.hpp"
    "src/packet/building/clientAgentInventoryRepairRequest.cpp"
    "src/packet/building/clientAgentInventoryRepairRequest.hpp"
    "src/packet/building/clientAgentInventoryStorageOpenRequest.cpp"
    "src/packet/building/clientAgentInventoryStorageOpenRequest.hpp"
    "src/packet/building/clientAgentOperatorRequest.cpp"
    "src/packet/building/clientAgentOperatorRequest.hpp"
    "src/packet/building/clientGatewayLoginIbuvAnswer.cpp"
    "src/packet/building/clientGatewayLoginIbuvAnswer.hpp"
    "src/packet/building/clientGatewayLoginRequest.cpp"
    "src/packet/building/clientGatewayLoginRequest.hpp"
    "src/packet/building/commonBuilding.cpp"
    "src/packet/building/commonBuilding.hpp"
    "src/packet/building/serverAgentChatUpdate.cpp"
    "src/packet/building/serverAgentChatUpdate.hpp"
    "src/packet/building/serverAgentEntityDespawn.cpp"
    "src/packet/building/serverAgentEntityDespawn.hpp"
    "src/packet/building/serverAgentEntitySpawn.cpp"
    "src/packet/building/serverAgentEntitySpawn.hpp"
    "src/packet/building/serverAgentEntityUpdateState.cpp"
    "src/packet/building/serverAgentEntityUpdateState.hpp"
    "src/packet/building/serverAgentInventoryOperationResponse.cpp"
    "src/packet/building/serverAgentInventoryOperationResponse.hpp"
    "src/packet/building/serverAgentSkillBegin.cpp"
    "src/packet/building/serverAgentSkillBegin.hpp"
    "src/packet/building/serverAgentSkillEnd.cpp"
    "src/packet/building/serverAgentSkillEnd.hpp"
    "src/packet/enums/packetEnums.cpp"
    "src/packet/enums/packetEnums.hpp"
    "src/packet/opcode.cpp"
    "src/packet/opcode.hpp"
    "src/packet/parsing/clientAgentActionCommandRequest.cpp"
    "src/packet/parsing/clientAgentActionCommandRequest.hpp"
    "src/packet/parsing/clientAgentActionDeselectRequest.cpp"
    "src/packet/parsing/clientAgentActionDeselectRequest.hpp"
    "src/packet/parsing/clientAgentActionSelectRequest.cpp"
    "src/packet/parsing/clientAgentActionSelectRequest.hpp"
    "src/packet/parsing/clientAgentActionTalkRequest.cpp"
    "src/packet/parsing/clientAgentActionTalkRequest.hpp"
    "src/packet/parsing/clientAgentCharacterMoveRequest.cpp"
    "src/packet/parsing/clientAgentCharacterMoveRequest.hpp"
    "src/packet/parsing/clientAgentChatRequest.cpp"
    "src/packet/parsing/clientAgentChatRequest.hpp"
    "src/packet/parsing/clientAgentInventoryItemUseRequest.cpp"
    "src/packet/parsing/clientAgentInventoryItemUseRequest.hpp"
    "src/packet/parsing/commonParsing.cpp"
    "src/packet/parsing/commonParsing.hpp"
    "src/packet/parsing/packetParser.cpp"
    "src/packet/parsing/packetParser.hpp"
    "src/packet/parsing/parsedPacket.cpp"
    "src/packet/parsing/parsedPacket.hpp"
    "src/packet/parsing/serverAgentActionCommandResponse.cpp"
    "src/packet/parsing/serverAgentActionCommandResponse.hpp"
    "src/packet/parsing/serverAgentActionDeselectResponse.cpp"
    "src/packet/parsing/serverAgentActionDeselectResponse.hpp"
    "src/packet/parsing/serverAgentActionSelectResponse.cpp"
    "src/packet/parsing/serverAgentActionSelectResponse.hpp"
    "src/packet/parsing/serverAgentActionTalkResponse.cpp"
    "src/packet/parsing/serverAgentActionTalkResponse.hpp"
    "src/packet/parsing/serverAgentBuffAdd.cpp"
    "src/packet/parsing/serverAgentBuffAdd.hpp"
    "src/packet/parsing/serverAgentBuffRemove.cpp"
    "src/packet/parsing/serverAgentBuffRemove.hpp"
    "src/packet/parsing/serverAgentCharacterData.cpp"
    "src/packet/parsing/serverAgentCharacterData.hpp"
    "src/packet/parsing/serverAgentChatUpdate.cpp"
    "src/packet/parsing/serverAgentChatUpdate.hpp"
    "src/packet/parsing/serverAgentCosData.cpp"
    "src/packet/parsing/serverAgentCosData.hpp"
    "src/packet/parsing/serverAgentEntityDespawn.cpp"
    "src/packet/parsing/serverAgentEntityDespawn.hpp"
    "src/packet/parsing/serverAgentEntityGroupSpawnData.cpp"
    "src/packet/parsing/serverAgentEntityGroupSpawnData.hpp"
    "src/packet/parsing/serverAgentEntityRemoveOwnership.cpp"
    "src/packet/parsing/serverAgentEntityRemoveOwnership.hpp"
    "src/packet/parsing/serverAgentEntitySpawn.cpp"
    "src/packet/parsing/serverAgentEntitySpawn.hpp"
    "src/packet/parsing/serverAgentEntitySyncPosition.cpp"
    "src/packet/parsing/serverAgentEntitySyncPosition.cpp"
    "src/packet/parsing/serverAgentEntityUpdateAngle.cpp"
    "src/packet/parsing/serverAgentEntityUpdateAngle.hpp"
    "src/packet/parsing/serverAgentEntityUpdateExperience.cpp"
    "src/packet/parsing/serverAgentEntityUpdateExperience.hpp"
    "src/packet/parsing/serverAgentEntityUpdateHwanLevel.cpp"
    "src/packet/parsing/serverAgentEntityUpdateHwanLevel.hpp"
    "src/packet/parsing/serverAgentEntityUpdateMovement.cpp"
    "src/packet/parsing/serverAgentEntityUpdateMovement.hpp"
    "src/packet/parsing/serverAgentEntityUpdateMoveSpeed.cpp"
    "src/packet/parsing/serverAgentEntityUpdateMoveSpeed.hpp"
    "src/packet/parsing/serverAgentEntityUpdatePoints.cpp"
    "src/packet/parsing/serverAgentEntityUpdatePoints.hpp"
    "src/packet/parsing/serverAgentEntityUpdatePosition.cpp"
    "src/packet/parsing/serverAgentEntityUpdatePosition.hpp"
    "src/packet/parsing/serverAgentEntityUpdateState.cpp"
    "src/packet/parsing/serverAgentEntityUpdateState.hpp"
    "src/packet/parsing/serverAgentEntityUpdateStatus.cpp"
    "src/packet/parsing/serverAgentEntityUpdateStatus.hpp"
    "src/packet/parsing/serverAgentGuildStorageData.cpp"
    "src/packet/parsing/serverAgentGuildStorageData.hpp"
    "src/packet/parsing/serverAgentInventoryItemUseResponse.cpp"
    "src/packet/parsing/serverAgentInventoryItemUseResponse.hpp"
    "src/packet/parsing/serverAgentInventoryOperationResponse.cpp"
    "src/packet/parsing/serverAgentInventoryOperationResponse.hpp"
    "src/packet/parsing/serverAgentInventoryRepairResponse.cpp"
    "src/packet/parsing/serverAgentInventoryRepairResponse.hpp"
    "src/packet/parsing/serverAgentInventoryStorageData.cpp"
    "src/packet/parsing/serverAgentInventoryStorageData.hpp"
    "src/packet/parsing/serverAgentInventoryUpdateDurability.cpp"
    "src/packet/parsing/serverAgentInventoryUpdateDurability.hpp"
    "src/packet/parsing/serverAgentInventoryUpdateItem.cpp"
    "src/packet/parsing/serverAgentInventoryUpdateItem.hpp"
    "src/packet/parsing/serverAgentSkillBegin.cpp"
    "src/packet/parsing/serverAgentSkillBegin.hpp"
    "src/packet/parsing/serverAgentSkillEnd.cpp"
    "src/packet/parsing/serverAgentSkillEnd.hpp"
    "src/packet/structures/packetInnerStructures.cpp"
    "src/packet/structures/packetInnerStructures.hpp"
    "src/packetLogger.cpp"
    "src/packetLogger.hpp"
    "src/packetProcessor.cpp"
    "src/packetProcessor.hpp"
    "src/pk2/characterData.cpp"
    "src/pk2/characterData.hpp"
    "src/pk2/gameData.cpp"
    "src/pk2/gameData.hpp"
    "src/pk2/itemData.cpp"
    "src/pk2/itemData.hpp"
    "src/pk2/levelData.cpp"
    "src/pk2/levelData.hpp"
    "src/pk2/magicOptionData.cpp"
    "src/pk2/magicOptionData.hpp"
    "src/pk2/parsing/navmeshParser.cpp"
    "src/pk2/parsing/navmeshParser.hpp"
    "src/pk2/parsing/regionInfoParser.cpp"
    "src/pk2/parsing/regionInfoParser.hpp"
    "src/pk2/refRegion.cpp"
    "src/pk2/refRegion.hpp"
    "src/pk2/regionInfo.cpp"
    "src/pk2/regionInfo.hpp"
    "src/pk2/shopData.cpp"
    "src/pk2/shopData.hpp"
    "src/pk2/skillData.cpp"
    "src/pk2/skillData.hpp"
    "src/pk2/teleportData.cpp"
    "src/pk2/teleportData.hpp"
    "src/pk2/textItemAndSkillData.cpp"
    "src/pk2/textItemAndSkillData.hpp"
    "src/pk2/textZoneNameData.cpp"
    "src/pk2/textZoneNameData.hpp"
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
    "src/state/entityTracker.cpp"
    "src/state/entityTracker.hpp"
    "src/state/machine/autoPotion.cpp"
    "src/state/machine/autoPotion.hpp"
    "src/state/machine/botting.cpp"
    "src/state/machine/botting.hpp"
    "src/state/machine/buyingItems.cpp"
    "src/state/machine/buyingItems.hpp"
    "src/state/machine/castSkill.cpp"
    "src/state/machine/castSkill.hpp"
    "src/state/machine/moveItemInInventory.cpp"
    "src/state/machine/moveItemInInventory.hpp"
    "src/state/machine/pickItem.cpp"
    "src/state/machine/pickItem.hpp"
    "src/state/machine/pickItemWithCos.cpp"
    "src/state/machine/pickItemWithCos.hpp"
    "src/state/machine/stateMachine.cpp"
    "src/state/machine/stateMachine.hpp"
    "src/state/machine/talkingToShopNpc.cpp"
    "src/state/machine/talkingToShopNpc.hpp"
    "src/state/machine/talkingToStorageNpc.cpp"
    "src/state/machine/talkingToStorageNpc.hpp"
    "src/state/machine/townlooping.cpp"
    "src/state/machine/townlooping.hpp"
    "src/state/machine/training.cpp"
    "src/state/machine/training.hpp"
    "src/state/machine/useItem.cpp"
    "src/state/machine/useItem.hpp"
    "src/state/machine/useReturnScroll.cpp"
    "src/state/machine/useReturnScroll.hpp"
    "src/state/machine/walking.cpp"
    "src/state/machine/walking.hpp"
    "src/state/self.cpp"
    "src/state/self.hpp"
    "src/state/skillEngine.cpp"
    "src/state/skillEngine.hpp"
    "src/state/worldState.cpp"
    "src/state/worldState.hpp"
    "src/storage/buybackQueue.cpp"
    "src/storage/buybackQueue.hpp"
    "src/storage/item.cpp"
    "src/storage/item.hpp"
    "src/storage/itemList.cpp"
    "src/storage/itemList.hpp"
    "src/storage/storage.cpp"
    "src/storage/storage.hpp"
    "src/type_id/categories.cpp"
    "src/type_id/categories.hpp"
    "src/type_id/typeCategory.cpp"
    "src/type_id/typeCategory.hpp"
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
    SilkroadLib
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
  SilkroadLib
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