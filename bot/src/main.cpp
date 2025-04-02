#include "hyperbot.hpp"

#include <pybind11/embed.h>

#include <tracy/Tracy.hpp>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/log/log_sink.h>
#include <absl/log/log_sink_registry.h>
#include <absl/strings/str_format.h>

#include <charconv>
#include <csignal>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

constexpr std::string_view kDefaultCharacterFlagValue{"__default_character__"};

ABSL_FLAG(int, v, -1, "Verbose logging level");
ABSL_FLAG(std::vector<std::string>, vmodule, {}, "Per-module verbose logging level");
ABSL_FLAG(std::string, character, std::string(kDefaultCharacterFlagValue), "Character to login");

using namespace std;

#if TRACY_ENABLE

// Override global new operator
void* operator new(size_t size) {
  void* ptr = malloc(size);
  if (!ptr) {
    throw std::bad_alloc();
  }
  TracyAlloc(ptr, size);
  return ptr;
}

// Override global new[] operator
void* operator new[](size_t size) {
  void* ptr = malloc(size);
  if (!ptr) {
    throw std::bad_alloc();
  }
  TracyAlloc(ptr, size);
  return ptr;
}

// Override global delete operator
void operator delete(void* ptr) noexcept {
  TracyFree(ptr);
  free(ptr);
}

// Override global delete[] operator
void operator delete[](void* ptr) noexcept {
  TracyFree(ptr);
  free(ptr);
}

#endif

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

  pybind11::scoped_interpreter guard(false);
  VLOG(1) << "Python interpreter instantiated";

  try {
    // Append the current source directory to sys.path so that we can later load any local python files. SOURCE_DIR is set from CMake.
    pybind11::module sys = pybind11::module::import("sys");
    const std::string sourceDir = std::string(SOURCE_DIR);
    sys.attr("path").cast<pybind11::list>().append(sourceDir);
    VLOG(1) << "Added \"" << sourceDir << "\" to python path";

    VLOG(1) << "Warming up JAX";
    // Warm-up JAX
    pybind11::module jax = pybind11::module::import("jax");
    // Force initialization by listing available devices.
    pybind11::object devices = jax.attr("devices")();
    // Optionally, log the devices.
    VLOG(1) << "JAX devices: " << std::string(pybind11::str(devices));
  } catch (const std::exception &ex) {
    LOG(ERROR) << absl::StreamFormat("Caught an exception while warming up JAX: \"%s\". Exiting.", ex.what());
    LOG(WARNING) << "If \"ModuleNotFoundError: No module named 'jax'\", you may need to source the virtual environment.";
    throw;
  }

  pybind11::gil_scoped_release release;
  VLOG(1) << "Python GIL released";

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

  static ColorLogSink colorLogSink;
  absl::AddLogSink(&colorLogSink);

  // Disable stderr logging, since we have installed our own log sink.
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfinity);
}