#ifndef CONFIG_CONFIG_HPP_
#define CONFIG_CONFIG_HPP_

#include "ui-proto/config.pb.h"

#include <filesystem>
#include <mutex>

namespace config {

class Config {
public:
  void initialize(const std::filesystem::path &pathToConfig);
  void save();
  void overwriteConfigProto(const proto::config::Config &configProto);
  proto::config::Config& configProto();
  const proto::config::Config& configProto() const;
  std::mutex& mutex() const;
private:
  inline static const std::string kConfigFileFilename{"config"};
  std::filesystem::path configFileFilePath_;
  mutable std::mutex mutex_;
  proto::config::Config configProto_;
};

} // namespace config

#endif // CONFIG_CONFIG_HPP_