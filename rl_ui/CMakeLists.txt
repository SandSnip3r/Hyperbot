project(rl_ui VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Charts Qml)
find_package(cppzmq CONFIG REQUIRED)

set(PROJECT_SOURCES
  main.cpp
  mainWindow.cpp
  mainWindow.hpp
  mainWindow.ui
)

qt_add_executable(rl_ui
  MANUAL_FINALIZATION
  ${PROJECT_SOURCES}
  hyperbotConnect.hpp hyperbotConnect.cpp hyperbotConnect.ui
  hyperbot.hpp hyperbot.cpp
  config.hpp config.cpp
  hyperbotConnectWorker.hpp hyperbotConnectWorker.cpp
  interactiveChartView.hpp interactiveChartView.cpp
  hyperbotSubscriberWorker.hpp hyperbotSubscriberWorker.cpp
  checkpointWidget.hpp checkpointWidget.cpp checkpointwidget.ui
  dashboardWidget.hpp dashboardWidget.cpp dashboardwidget.ui
  characterDetailDialog.hpp characterDetailDialog.cpp characterDetailDialog.ui
  barStyles.hpp barStyles.cpp
  draggableTableWidget.hpp draggableTableWidget.cpp
  textureToQImage.hpp
)

target_link_libraries(rl_ui PRIVATE
  absl::flags
  absl::flags_parse
  absl::log
  absl::log_initialize
  absl::strings
  cppzmq
  Qt6::Widgets
  Qt6::Charts
  Qt6::Qml
  silkroad_lib
  ui_proto
  gli
)

set_target_properties(rl_ui PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS rl_ui
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

qt_finalize_executable(rl_ui)
