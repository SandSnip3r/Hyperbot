#include "config.hpp"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <fstream>
#include <stdexcept>

namespace config {

void Config::initialize(const std::filesystem::path &pathToConfig) {
  configFileFilePath_ = pathToConfig / kConfigFileFilename;

  std::ios_base::openmode fileOpenMode = std::ios::in;
  if constexpr (kProtobufSavedAsBinary_) {
    // Reading file as binary
    fileOpenMode |= std::ios::binary;
  }

  std::ifstream configFileIn(configFileFilePath_, fileOpenMode);
  if (!configFileIn) {
    // Config file does not exist, create a new one with our default constructed proto
    save();
    return;
  }

  // Config file exists, parse
  bool success;
  if constexpr (kProtobufSavedAsBinary_) {
    // Try to read an existing open config file as binary
    success = configProto_.ParseFromIstream(&configFileIn);
  } else {
    // Try to read an existing open config file as string
    google::protobuf::io::IstreamInputStream pbIStream(&configFileIn);
    success = google::protobuf::TextFormat::Parse(&pbIStream, &configProto_);
  }

  if (!success) {
    throw std::runtime_error("Config file open, but could not parse");
  }
}

void Config::save() {
  std::ios_base::openmode fileOpenMode = std::ios::out | std::ios::trunc;
  if constexpr (kProtobufSavedAsBinary_) {
    // Writing file as binary
    fileOpenMode |= std::ios::binary;
  }

  std::ofstream configFileOut(configFileFilePath_, fileOpenMode);
  if (!configFileOut) {
    throw std::runtime_error("Could not open config file for writing");
  }

  bool success;
  if constexpr (kProtobufSavedAsBinary_) {
    // Try to write the config as binary to the open file
    success = configProto_.SerializeToOstream(&configFileOut);
  } else {
    // Try to write the config as a string to the open file
    google::protobuf::io::OstreamOutputStream pbOStream(&configFileOut);
    success = google::protobuf::TextFormat::Print(configProto_, &pbOStream);
  }

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

proto::config::CharacterConfig* Config::getCharacterConfig(absl::string_view characterName) {
  const auto it = std::find_if(configProto_.mutable_character_configs()->begin(), configProto_.mutable_character_configs()->end(), [&characterName](const auto &characterConfig) {
    return characterConfig.character_name() == characterName;
  });
  if (it == configProto_.mutable_character_configs()->end()) {
    return nullptr;
  }
  return &(*it);
}

const proto::config::CharacterConfig* Config::getCharacterConfig(absl::string_view characterName) const {
  const auto it = std::find_if(configProto_.character_configs().cbegin(), configProto_.character_configs().cend(), [&characterName](const auto &characterConfig) {
    return characterConfig.character_name() == characterName;
  });
  if (it == configProto_.character_configs().cend()) {
    return nullptr;
  }
  return &(*it);
}

std::optional<LoginInfo> Config::getLoginInfo(absl::string_view characterName) const {
  const auto *characterConfig = getCharacterConfig(characterName);
  if (characterConfig == nullptr) {
    LOG(WARNING) << absl::StrFormat("Cannot get login info for unknown character \"%s\"", characterName);
    return {};
  }
  return LoginInfo{characterConfig->username(), characterConfig->password()};
}

} // namespace config