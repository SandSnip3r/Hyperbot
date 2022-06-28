#include "bot.hpp"

Bot::Bot(const config::CharacterLoginData &loginData,
         const pk2::GameData &gameData,
         broker::PacketBroker &broker) :
      loginData_(loginData),
      gameData_(gameData),
      broker_(broker) {
  eventBroker_.run();
  userInterface_.run();
}