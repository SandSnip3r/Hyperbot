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
  proxy_.Stop();
  // loader_.killClient();
}

void Session::start() {
  auto port = proxy_.getOurListeningPort();
  loader_.startClient(port); //throws if problem starting client & injecting
  proxy_.start(); //throws if socket issues, blocks
}