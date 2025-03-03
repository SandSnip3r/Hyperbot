#include "hyperbot.hpp"

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/log/log_sink.h>
#include <absl/log/log_sink_registry.h>
#include <absl/strings/str_format.h>

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

constexpr std::string_view kDefaultCharacterFlagValue{"__default_character__"};

ABSL_FLAG(int, v, -1, "Verbose logging level");
ABSL_FLAG(std::vector<std::string>, vmodule, {}, "Per-module verbose logging level");
ABSL_FLAG(std::string, character, std::string(kDefaultCharacterFlagValue), "Character to login");

using namespace std;

void initializeLogging();

class ColorLogSink : public absl::LogSink {
public:
  void Send(const absl::LogEntry& entry) override {
    constexpr string_view kRedEscapeCode{"\033[31m"};
    constexpr string_view kYellowEscapeCode{"\033[33m"};
    constexpr string_view kResetEscapeCode{"\033[0m"};
    if (entry.log_severity() == absl::LogSeverity::kError) {
      cerr << kRedEscapeCode << entry.text_message_with_prefix() << kResetEscapeCode << endl;
    } else if (entry.log_severity() == absl::LogSeverity::kWarning) {
      cerr << kYellowEscapeCode << entry.text_message_with_prefix() << kResetEscapeCode << endl;
    } else {
      cerr << entry.text_message_with_prefix_and_newline() << flush;
    }
  }
};

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  initializeLogging();

  cerr << "\033[30;42m" << R"(     __  __                      __          __  )" << "\033[0m\n";
  cerr << "\033[30;42m" << R"(    / / / /_  ______  ___  _____/ /_  ____  / /_ )" << "\033[0m\n";
  cerr << "\033[30;42m" << R"(   / /_/ / / / / __ \/ _ \/ ___/ __ \/ __ \/ __/ )" << "\033[0m\n";
  cerr << "\033[30;42m" << R"(  / __  / /_/ / /_/ /  __/ /  / /_/ / /_/ / /_   )" << "\033[0m\n";
  cerr << "\033[30;42m" << R"( /_/ /_/\__, / .___/\___/_/  /_.___/\____/\__/   )" << "\033[0m\n";
  cerr << "\033[30;42m" << R"(       /____/_/                                  )" << "\033[0m\n";
  cerr << flush;

  VLOG(1) << "Abseil logging initialized";

  try {
    Hyperbot hyperbot;
    hyperbot.run();
  } catch (const std::exception &ex) {
    LOG(ERROR) << absl::StreamFormat("Caught an exception while running Hyperbot: \"%s\". Exiting.", ex.what());
    return 1;
  }
  return 0;
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

  // Set everything to log to stderr.
  // absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  static ColorLogSink colorLogSink;
  absl::AddLogSink(&colorLogSink);
}