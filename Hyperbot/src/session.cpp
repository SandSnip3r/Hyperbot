#include "session.hpp"

Session::Session(const pk2::GameData &gameData,
                 std::string_view clientPath,
                 broker::EventBroker &eventBroker) :
    gameData_(gameData),
    clientPath_(clientPath),
    eventBroker_(eventBroker) {
}

Session::~Session() {
  proxy_.stop();
  proxyThread_.join();
  // loader_.killClient();
}

void Session::initialize() {
  bot_.initialize();
}

void Session::runAsync() {
  loader_.startClient(proxy_.getOurListeningPort()); //throws if problem starting client & injecting
  proxyThread_ = std::thread(std::bind(&Proxy::run, &proxy_));
  // proxy_.run(); //throws if socket issues, blocks
}

const state::WorldState& Session::getWorldState() const {
  return bot_.worldState();
}