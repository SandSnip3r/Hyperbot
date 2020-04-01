#include "session.hpp"

Session::Session(const pk2::GameData &gameData,
                 const std::experimental::filesystem::v1::path &kSilkroadDirectoryPath,
                 const config::CharacterLoginData &loginData) :
    gameData_(gameData),
    kSilkroadDirectoryPath_(kSilkroadDirectoryPath),
    loginData_(loginData) {
  //
}

Session::~Session() {
  proxy_.Stop();
  // loader_.killClient();
}

void Session::start() {
  loader_.startClient(); //throws if problem starting client & injecting
  proxy_.start(); //throws if socket issues, blocks
}