#include "session.hpp"

Session::Session(const pk2::GameData &gameData,
                 const std::filesystem::path &kSilkroadDirectoryPath,
                 const config::CharacterLoginData &loginData) :
    gameData_(gameData),
    kSilkroadDirectoryPath_(kSilkroadDirectoryPath),
    loginData_(loginData)/* ,
    pathfinder_() */ {
  //
}

Session::~Session() {
  proxy_.stop();
  // loader_.killClient();
}

void Session::run() {
  eventBroker_.runAsync();
  auto port = proxy_.getOurListeningPort();
  loader_.startClient(port); //throws if problem starting client & injecting
  proxy_.run(); //throws if socket issues, blocks
}