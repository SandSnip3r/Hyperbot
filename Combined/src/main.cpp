#include "session.hpp"
#include "silkroadConnection.hpp"

#include "broker/eventBroker.hpp"
#include "config/configData.hpp"
#include "config/iniReader.hpp"
#include "pk2/gameData.hpp"
#include "ui/userInterface.hpp"
#include "ui-proto/config.pb.h"

#include "../../common/Common.h"
#include "../../common/pk2/divisionInfo.hpp"
#include "../../common/pk2/parsing/parsing.hpp"

#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

using namespace std;
namespace fs = std::filesystem;

fs::path getAppDataDirectory();

class Hyperbot {
public:
  Hyperbot(const std::optional<std::string> &characterToLogin) : characterToLogin_(characterToLogin) {}
  void run() {
    const auto appDataDirectory = getAppDataDirectory();
    config_.initialize(appDataDirectory);
    if (characterToLogin_) {
      // TODO: This will result in overwriting of the original config file if this is saved.
      LOG(INFO) << "Given a character to log into: \"" << *characterToLogin_ << "\". Overwritting config.";
      config_.configProto().set_character_to_login(*characterToLogin_);
    } else {
      LOG(INFO) << "Using default character to login";
    }

    subscribeToEvents();
    ui::UserInterface userInterface{gameData_, eventBroker_}; // TODO: Remove gameData_ from user interface
    userInterface.initialize();

    eventBroker_.runAsync();
    userInterface.runAsync();

    tryInitializeGameData();
    {
      std::unique_lock<std::mutex> gameDataLock(gameDataMutex_);
      if (!gameDataReady_) {
        // Game data hasn't yet been parsed, need to wait until the config is updated to contain a sro_client path which has parsable game data
        LOG(INFO) << "Waiting until game data is parsed";
        gameDataParsedConditionVariable_.wait(gameDataLock, [this](){
          return gameDataReady_;
        });
        LOG(INFO) << "Done parsing game data";
      }
    }

    try {
      Session session{gameData_, config_, eventBroker_};
      session.initialize();
      userInterface.setWorldState(session.getWorldState());
      userInterface.broadcastLaunch();
      session.run();
    } catch (const std::exception &ex) {
      LOG(INFO) << "Error while running session: \"" << ex.what() << '"';
    }
  }

private:
  std::optional<std::string> characterToLogin_;
  config::Config config_;
  broker::EventBroker eventBroker_;
  bool waitingForClientPath_{true};
  pk2::GameData gameData_;
  bool gameDataReady_{false};
  std::condition_variable gameDataParsedConditionVariable_;
  std::mutex gameDataMutex_;

  void subscribeToEvents() {
    auto eventHandleFunction = std::bind(&Hyperbot::handleEvent, this, std::placeholders::_1);
    eventBroker_.subscribeToEvent(event::EventCode::kNewConfigReceived, eventHandleFunction);
  }

  void handleEvent(const event::Event *event) {
    if (event->eventCode == event::EventCode::kNewConfigReceived) {
      const auto &castedEvent = dynamic_cast<const event::NewConfigReceived&>(*event);
      handleNewConfigReceived(castedEvent);
    }
  }

  void handleNewConfigReceived(const event::NewConfigReceived &event) {
    {
      std::unique_lock<std::mutex> configLock(config_.mutex());
      config_.overwriteConfigProto(event.config);
    }
    eventBroker_.publishEvent(event::EventCode::kConfigUpdated);

    bool gameDataIsAlreadyParsed{false};
    {
      std::unique_lock<std::mutex> gameDataLock(gameDataMutex_);
      gameDataIsAlreadyParsed = gameDataReady_;
    }
    if (gameDataIsAlreadyParsed) {
      // We do not support changing the sro_client path once we've already successfully parsed the game data
      return;
    }
    tryInitializeGameData();
  }

  void tryInitializeGameData() {
    std::string clientPath;
    {
      std::unique_lock<std::mutex> configLock(config_.mutex());
      if (!config_.configProto().has_client_path()) {
        LOG(INFO) << "Config does not contain a client path. Nothing to do";
        return;
      }
      clientPath = config_.configProto().client_path();
    }
    try {
      std::unique_lock<std::mutex> gameDataLock(gameDataMutex_);

      // Overwrite old one, just in case we already tried to parse one and it failed
      // gameData_ = pk2::GameData(); // TODO: Modify GameData so that it is in a good state if parsing fails

      gameData_.parseSilkroadFiles(clientPath);
      gameDataReady_ = true;
      gameDataParsedConditionVariable_.notify_one();
    } catch (const std::exception &ex) {
      LOG(INFO) << "Failed to parse Silkroad game data: \"" << ex.what() << "\"";
      // If we fail to parse the game data, we don't know that the GameData object is in a good state
      //  Throw an error so we don't end up here again
      throw std::runtime_error("We failed to parse the Silkroad game data");
    }
  }
};

int main(int argc, char **argv) {
  // Initialize abseil logging.
  absl::InitializeLog();
  // Ideally, this setting of the threshold and vlog level should come from the commandline, but something about our cmake setup results in the relevant flags not being linked into this binary.
  absl::SetGlobalVLogLevel(10);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  std::optional<std::string> characterToLogin;
  if (argc > 1) {
    characterToLogin = argv[1];
    LOG(INFO) << "Want to log in character \"" << *characterToLogin << "\"";
  }
  Hyperbot hyperbot(characterToLogin);
  try {
    hyperbot.run();
  } catch (const std::exception &ex) {
    LOG(INFO) << "Caught an exception; exiting. \"" << ex.what() << '"';
    return 1;
  }
  return 0;
}

fs::path getAppDataDirectory() {
  const auto appDataDirectoryPath = getAppDataPath();
  if (appDataDirectoryPath.empty()) {
    throw std::runtime_error("Unable to find %APPDATA%\n");
  }
  // Make sure the directory exists
  if (!fs::exists(appDataDirectoryPath)) {
    fs::create_directory(appDataDirectoryPath);
  }
  return appDataDirectoryPath;
}