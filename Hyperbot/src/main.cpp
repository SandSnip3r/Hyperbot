#include "hyperbot.hpp"

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <charconv>
#include <string>
#include <string_view>
#include <vector>

constexpr std::string_view kDefaultCharacterFlagValue{"__default_character__"};

ABSL_FLAG(int, v, -1, "Verbose logging level");
ABSL_FLAG(std::vector<std::string>, vmodule, {}, "Per-module verbose logging level");
ABSL_FLAG(std::string, character, std::string(kDefaultCharacterFlagValue), "Character to login");

using namespace std;

void initializeLogging();

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  initializeLogging();

  try {
    Hyperbot hyperbot;
    hyperbot.run();
  } catch (const std::exception &ex) {
    LOG(INFO) << absl::StreamFormat("Caught an exception while running Hyperbot: \"%s\". Exiting.", ex.what());
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
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
}