#include "logging.hpp"
#include "session.hpp"

Session::Session(const pk2::GameData &gameData,
                 const config::Config &config,
                 broker::EventBroker &eventBroker) :
    gameData_(gameData),
    config_(config),
    eventBroker_(eventBroker) {
}

Session::~Session() {
  proxy_.stop();
  // loader_.killClient();
}

void Session::initialize() {
  bot_.initialize();
}

void Session::run() {
  bot_.run(); // TODO: We should get rid of this function once we move the world state out of the bot.
  loader_.startClient(proxy_.getOurListeningPort()); //throws if problem starting client & injecting
  proxy_.run(); //throws if socket issues, blocks
}

const state::WorldState& Session::getWorldState() const {
  return bot_.worldState();
}