#include "clientManager.hpp"

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <charconv>
#include <iostream>
#include <thread>

ABSL_FLAG(int, v, -1, "Verbose logging level");
ABSL_FLAG(std::vector<std::string>, vmodule, {}, "Per-module verbose logging level");
ABSL_FLAG(std::string, ip_address, "127.0.0.1", "IP address of Hyperbot");
ABSL_FLAG(int32_t, protobufPort, 2235, "Port of Hyperbot");
ABSL_FLAG(int32_t, gatewayPort, 15779, "Original port of the Silkroad Gateway Server");
ABSL_FLAG(int32_t, agentPort, 15884, "Original port of the Silkroad Agent Server");
ABSL_FLAG(std::string, client_path, "C:\\dev\\VSRO_Client", "Path to sro_client.exe");

void initializeLogging();

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);
  initializeLogging();

  const std::string ipAddress = absl::GetFlag(FLAGS_ip_address);
  const int32_t protobufPort = absl::GetFlag(FLAGS_protobufPort);
  const int32_t gatewayPort = absl::GetFlag(FLAGS_gatewayPort);
  const int32_t agentPort = absl::GetFlag(FLAGS_agentPort);
  const std::string client_path = absl::GetFlag(FLAGS_client_path);
  VLOG(2) << "Parsed IP address: \"" << ipAddress << "\" and protobufPort: " << protobufPort;
  VLOG(2) << "Parsed client path: \"" << client_path << "\"";

  try {
    ClientManager clientManager{ipAddress, protobufPort, gatewayPort, agentPort, client_path};
    clientManager.run();
  } catch (const std::exception &ex) {
    LOG(ERROR) << absl::StreamFormat("Error running ClientManager: \"%s\"", ex.what());
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
      absl::SetVLogLevel(absl::string_view(module_name.data(), module_name.size()), level);
    }
  }

  // Set everything to log to stderr.
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
}