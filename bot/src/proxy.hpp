#ifndef PROXY_HPP_
#define PROXY_HPP_

#include "packetLogger.hpp"
#include "silkroadConnection.hpp"
#include "broker/packetBroker.hpp"
#include "pk2/gameData.hpp"
#include "shared/silkroad_security.h"

#include <silkroad_lib/pk2/divisionInfo.hpp>

#include <boost/make_shared.hpp>
#include <boost/asio.hpp>

#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

//Networking class (handles connections)
class Proxy {
public:
	Proxy(const pk2::GameData &gameData, broker::PacketBroker &broker, uint16_t port=0);
	~Proxy();
	void inject(const PacketContainer &packet, const PacketContainer::Direction direction);
  void runAsync();
  uint16_t getOurListeningPort() const;
  void blockOpcode(packet::Opcode opcode);
  void unblockOpcode(packet::Opcode opcode);
  bool blockingOpcode(packet::Opcode opcode) const;

	//Stops all networking objects
	void stop();
private:
  uint16_t ourListeningPort_;
  uint16_t gatewayPort_;
  const sro::pk2::DivisionInfo divisionInfo_;
  broker::PacketBroker &packetBroker_;
  std::string gatewayAddress_;
  std::string agentIP_;
  uint16_t agentPort_{0};
  std::set<uint16_t> blockedOpcodes_;
  bool connectToAgent{false};
  boost::asio::io_service ioService_;
  const int kPacketProcessDelayMs{10};
	PacketLogger packetLogger{"C:\\Users\\Victor\\Documents\\Development\\packet-logs\\"};
  std::optional<PacketContainer> characterInfoPacketContainer_, groupSpawnPacketContainer_, storagePacketContainer_, guildStoragePacketContainer_;
  std::thread thr_;

	//Accepts TCP connections
	boost::asio::ip::tcp::acceptor acceptor;

	//Packet processing timer
	boost::shared_ptr<boost::asio::deadline_timer> timer;

	//Silkroad connections
	SilkroadConnection clientConnection{ioService_};
	SilkroadConnection serverConnection{ioService_};

	//Starts accepting new connections
	void PostAccept(uint32_t count = 1);

	//Handles new connections
	void HandleAccept(boost::shared_ptr<boost::asio::ip::tcp::socket> s, const boost::system::error_code & error);

	void ProcessPackets(const boost::system::error_code & error);

  void run();
};

#endif