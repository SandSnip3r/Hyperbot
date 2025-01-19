#include "config.hpp"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <absl/log/log.h>

#include <fstream>

Config::Config(const std::filesystem::path &filePath) : filePath_(filePath) {}

void Config::load() {
  LOG(INFO) << "Looking for config at " << filePath_;
  std::ios_base::openmode fileOpenMode = std::ios::in;
  if constexpr (kProtobufSavedAsBinary_) {
    // Reading file as binary
    fileOpenMode |= std::ios::binary;
    LOG(INFO) << "Reading config as binary";
  }

  std::ifstream configFileIn(filePath_, fileOpenMode);
  if (!configFileIn) {
    LOG(INFO) << "Config does not exist. Constructing and using empty config";
    makeAndSaveDefaultConfig();
    return;
  }

  // ServerConfig file exists, parse
  bool success;
  if constexpr (kProtobufSavedAsBinary_) {
    // Try to read an existing open config file as binary
    success = proto_.ParseFromIstream(&configFileIn);
  } else {
    // Try to read an existing open config file as string
    google::protobuf::io::IstreamInputStream pbIStream(&configFileIn);
    success = google::protobuf::TextFormat::Parse(&pbIStream, &proto_);
  }

  LOG(INFO) << "Successfully found config file";

  if (!success) {
    LOG(WARNING) << "Config file open, but could not parse. Overwriting with default config";
    makeAndSaveDefaultConfig();
  }
}

void Config::save() const {
  LOG(INFO) << "Saving config:\n" << proto_.DebugString();
  std::ios_base::openmode fileOpenMode = std::ios::out | std::ios::trunc;
  if constexpr (kProtobufSavedAsBinary_) {
    LOG(INFO) << "  Saving as binary";
    // Writing file as binary
    fileOpenMode |= std::ios::binary;
  }

  std::ofstream configFileOut(filePath_, fileOpenMode);
  if (!configFileOut) {
    throw std::runtime_error("Could not open config file for writing");
  }

  bool success;
  if constexpr (kProtobufSavedAsBinary_) {
    // Try to write the config as binary to the open file
    success = proto_.SerializeToOstream(&configFileOut);
  } else {
    // Try to write the config as a string to the open file
    google::protobuf::io::OstreamOutputStream pbOStream(&configFileOut);
    success = google::protobuf::TextFormat::Print(proto_, &pbOStream);
  }

  if (!success) {
    throw std::runtime_error("Could not write config to file");
  }

  LOG(INFO) << "Successfully saved config";
}

void Config::makeAndSaveDefaultConfig() {
  proto_ = proto::rl_ui_config::Config();
  proto_.set_ip_address("127.0.0.1");
  proto_.set_port(2235);
  save();
}