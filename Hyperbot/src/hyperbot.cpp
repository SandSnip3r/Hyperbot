#include "hyperbot.hpp"

#include "helpers.hpp"
#include "session.hpp"
#include "ui/userInterface.hpp"

void Hyperbot::run() {
  parseConfig();

  ui::UserInterface userInterface{gameData_, eventBroker_};
  userInterface.initialize();

  eventBroker_.runAsync();
  userInterface.runAsync();

  initializeGameData();

  try {
    Session session{gameData_, serverConfig_.clientPath(), eventBroker_};
    Session session2{gameData_, serverConfig_.clientPath(), eventBroker_};
    session.initialize();
    session2.initialize();
    userInterface.setWorldState(session.getWorldState());
    userInterface.broadcastLaunch();
    session.runAsync();
    session2.runAsync();
    while (1) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  } catch (const std::exception &ex) {
    LOG(INFO) << "Error while running session: \"" << ex.what() << '"';
  }
}

void Hyperbot::parseConfig() {
  const auto appDataDirectory = helpers::getAppDataDirectory();
  serverConfig_.initialize(appDataDirectory);
}

void Hyperbot::initializeGameData() {
  try {
    gameData_.parseSilkroadFiles(serverConfig_.clientPath());
  } catch (const std::exception &ex) {
    throw std::runtime_error(absl::StrFormat("Failed to parse the Silkroad game data: \"%s\"", ex.what()));
  }
}