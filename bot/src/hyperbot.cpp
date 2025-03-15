#include "hyperbot.hpp"

#include "clientManagerInterface.hpp"
#include "helpers.hpp"
#include "rl/trainingManager.hpp"
#include "session.hpp"
#include "state/worldState.hpp"
#include "ui/rlUserInterface.hpp"

#include <absl/log/log.h>

#include <stdexcept>

void Hyperbot::run() {
  VLOG(1) << "Running Hyperbot";
  parseConfig();

  zmq::context_t context;

  ui::RlUserInterface rlUserInterface{context, eventBroker_};
  rlUserInterface.initialize();
  ClientManagerInterface clientManagerInterface(context, eventBroker_);

  clientManagerInterface.runAsync();
  eventBroker_.runAsync();
  rlUserInterface.runAsync();

  initializeGameData();

  // Create a single WorldState to be shared across all sessions.
  state::WorldState worldState{gameData_, eventBroker_};

  rl::TrainingManager trainingManager{gameData_, eventBroker_, worldState, clientManagerInterface};
  trainingManager.run();
}

Hyperbot::~Hyperbot() {
  VLOG(1) << "Destructing Hyperbot";
}

void Hyperbot::parseConfig() {
  const auto appDataDirectory = helpers::getAppDataDirectory();
  VLOG(2) << absl::StreamFormat("Parsing Hyperbot config at \"%s\"", appDataDirectory);
  serverConfig_.initialize(appDataDirectory);
  VLOG(2) << "Finished parsing config";
}

void Hyperbot::initializeGameData() {
  try {
    gameData_.parseSilkroadFiles(serverConfig_.clientPath());
  } catch (const std::exception &ex) {
    throw std::runtime_error(absl::StrFormat("Failed to parse the Silkroad game data: \"%s\"", ex.what()));
  }
}