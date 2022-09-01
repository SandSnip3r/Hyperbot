cmake_minimum_required(VERSION 3.5)

project(hyperbot-ui VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FindProtobuf)
add_subdirectory(proto)
add_subdirectory(SilkroadLib)

find_package(cppzmq CONFIG REQUIRED)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets LinguistTools)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets LinguistTools)

set(TS_FILES hyperbot-ui_en_US.ts)

set(PROJECT_SOURCES
  main.cpp
  characterData.cpp
  characterData.hpp
  eventHandler.cpp
  eventHandler.hpp
  experienceProgressBar.cpp
  experienceProgressBar.hpp
  itemListWidget.cpp
  itemListWidget.hpp
  itemListWidgetItem.cpp
  itemListWidgetItem.hpp
  mainwindow.cpp
  mainwindow.h
  packetListWidgetItem.cpp
  packetListWidgetItem.hpp
  reinjectablePacketListWidget.cpp
  reinjectablePacketListWidget.hpp
  requester.cpp
  requester.hpp
  mainwindow.ui
  ${TS_FILES}
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(hyperbot-ui
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET hyperbot-ui APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation

    qt_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
else()
    if(ANDROID)
        add_library(hyperbot-ui SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(hyperbot-ui
            ${PROJECT_SOURCES}
        )
    endif()

    qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
endif()

target_link_libraries(hyperbot-ui
PRIVATE
  Qt${QT_VERSION_MAJOR}::Widgets
  cppzmq
  cppzmq-static
  proto
  ${PROTOBUF_LIBRARIES}
  SilkroadLib
)

set_target_properties(hyperbot-ui PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(hyperbot-ui)
endif()
