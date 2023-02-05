#include "config.hpp"

#include <fstream>
#include <stdexcept>

namespace config {

void Config::initialize(const std::filesystem::path &pathToConfig) {
  configFileFilePath_ = pathToConfig / kConfigFileFilename;

  // Try to read an existing config file
  std::ifstream configFileIn(configFileFilePath_, std::ios::binary);
  if (configFileIn) {
    // Config file exists, parse
    bool success = configProto_.ParseFromIstream(&configFileIn);
    if (!success) {
      throw std::runtime_error("Config file open, but could not parse");
    }
    return;
  }

  // Config file does not exist, create a new one with our default constructed proto
  save();
}

void Config::save() {
  std::ofstream configFileOut(configFileFilePath_, std::ios::binary | std::ios::trunc);
  if (!configFileOut) {
    throw std::runtime_error("Could not open config file for writing");
  }
  bool success = configProto_.SerializeToOstream(&configFileOut);
  if (!success) {
    throw std::runtime_error("Could not write config to file");
  }
}

void Config::overwriteConfigProto(const proto::config::Config &configProto) {
  configProto_ = configProto;
  save();
}

proto::config::Config& Config::configProto() {
  return configProto_;
}

const proto::config::Config& Config::configProto() const {
  return configProto_;
}

std::mutex& Config::mutex() const {
  return mutex_;
}

} // namespace config