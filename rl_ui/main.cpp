#include "config.hpp"
#include "hyperbot.hpp"
#include "mainWindow.hpp"

#include <silkroad_lib/file_util.hpp>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/log/log_sink.h>
#include <absl/log/log_sink_registry.h>
#include <absl/strings/str_format.h>

#include <QApplication>
#include <QStyleFactory>

#include <charconv>
#include <iostream>

ABSL_FLAG(int, v, -1, "Verbose logging level");
ABSL_FLAG(std::vector<std::string>, vmodule, {}, "Per-module verbose logging level");

void initializeLogging();
std::filesystem::path getAppDataDirectory();

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);
  initializeLogging();

  VLOG(1) << "Looking for config...";
  const auto appDataDirectory = getAppDataDirectory();
  constexpr std::string_view kConfigFileFilename{"rl_ui_config"};
  std::filesystem::path configFileFilePath = appDataDirectory / kConfigFileFilename;
  Config config(configFileFilePath);
  config.load();

  Hyperbot bot;

  QApplication a(argc, argv);
  a.setStyle(QStyleFactory::create("Fusion"));
  // Note: QApplication must be constructed before any QWidget.
  MainWindow *mainWindow = new MainWindow(std::move(config), bot);
  mainWindow->show();
  return a.exec();
}

void initializeLogging() {
  absl::InitializeLog();

  // Set log levels from flags.
  absl::SetGlobalVLogLevel(absl::GetFlag(FLAGS_v));

  // Set per-module vlog levels.
  const auto vmodules = absl::GetFlag(FLAGS_vmodule);
  for (std::string_view module : vmodules) {
    const std::string::size_type equals_pos = module.find('=');
    if (equals_pos != std::string::npos) {
      std::string_view module_name = module.substr(0, equals_pos);
      std::string_view module_level = module.substr(equals_pos + 1);
      int level;
      const auto parseResult = std::from_chars(module_level.data(), module_level.data()+module_level.size(), level);
      if (parseResult.ec != std::errc{}) {
        const auto errorCode = std::make_error_code(parseResult.ec);
        throw std::runtime_error(absl::StrFormat("Could not parse integer from vmodule flag: \"%s\". Error is \"%s\"", module, errorCode.message()));
      }
      absl::SetVLogLevel(module_name, level);
    }
  }

  // QtCreator has a bug where every line of output from Abseil log statements is printed twice in the Application Output window.
  //  Bug: https://bugreports.qt.io/browse/QTCREATORBUG-30163
  constexpr bool kWorkaroundForQtCreatorBug{true};
  if constexpr (kWorkaroundForQtCreatorBug) {
    // To work around this bug, don't use the built-in logging mechanism, instead create our own LogSink which does the outputting for us.
    class MyLogSink : public absl::LogSink {
    public:
      void Send(const absl::LogEntry& entry) override {
        std::cout << entry.text_message_with_prefix_and_newline() << std::flush;
      }
    };
    absl::LogSink *mySink = new MyLogSink;
    absl::AddLogSink(mySink);
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfinity);
  } else {
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  }
}

std::filesystem::path getAppDataDirectory() {
  const auto appDataDirectoryPath = sro::file_util::getAppDataPath();
  if (appDataDirectoryPath.empty()) {
    throw std::runtime_error("Unable to find %APPDATA%\n");
  }
  // Make sure the directory exists
  if (!std::filesystem::exists(appDataDirectoryPath)) {
    std::filesystem::create_directory(appDataDirectoryPath);
  }
  return appDataDirectoryPath;
}
