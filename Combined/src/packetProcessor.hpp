#ifndef PACKETPROCESSOR_HPP_
#define PACKETPROCESSOR_HPP_

#include "broker/eventBroker.hpp"
#include "broker/packetBroker.hpp"
#include "packet/parsing/packetParser.hpp"
#include "pk2/gameData.hpp"
#include "state/entity.hpp"
#include "state/self.hpp"
#include "ui/userInterface.hpp"

/*  PacketProcessor
 *  As packets come in, this class will update the state
 */
class PacketProcessor {
public:
  PacketProcessor(state::Entity &entityState,
                  state::Self &selfState,
                  broker::PacketBroker &brokerSystem,
                  broker::EventBroker &eventBroker,
                  ui::UserInterface &userInterface,
                  const packet::parsing::PacketParser &packetParser,
                  const pk2::GameData &gameData);

  bool handlePacket(const PacketContainer &packet) const;
private:
  state::Entity &entityState_;
  state::Self &selfState_;
  broker::PacketBroker &broker_;
  broker::EventBroker &eventBroker_;
  ui::UserInterface &userInterface_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::GameData &gameData_;

  void subscribeToPackets();

  // Packet handle functions
  // From LoginModule
  void serverListReceived(const packet::parsing::ParsedLoginServerList &packet) const;
  void loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet) const;
  void loginClientInfoReceived(const packet::parsing::ParsedLoginClientInfo &packet) const;
  bool unknownPacketReceived(const packet::parsing::ParsedUnknown &packet) const;
  void serverAuthReceived(const packet::parsing::ParsedServerAuthResponse &packet) const;
  void charListReceived(const packet::parsing::ParsedServerAgentCharacterSelectionActionResponse &packet) const;
  void charSelectionJoinResponseReceived(const packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse &packet) const;
};

#endif // PACKETPROCESSOR_HPP_