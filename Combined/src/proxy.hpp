#include "packetLogger.hpp"
#include "silkroadConnection.hpp"
#include "shared/silkroad_security.h"

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/asio.hpp>

#include <functional>
#include <unordered_map>

#ifndef PROXY_HPP_
#define PROXY_HPP_

//Networking class (handles connections)
class Proxy {
public:
	Proxy(std::function<bool(const PacketContainer&, PacketContainer::Direction)> packetHandlerFunction, uint16_t port);

	~Proxy();

	void inject(PacketContainer &packet, PacketContainer::Direction direction);
  void start();

	//Stops all networking objects
	void Stop();
private:
  std::string agentIP_;
  uint16_t agentPort_{0};
  std::unordered_map<uint16_t, bool> blockedOpcodes_;
  bool connectToAgent{false};
  boost::asio::io_service ioService_;
  const int kPacketProcessDelayMs{10};
	std::function<bool(const PacketContainer&, PacketContainer::Direction)> packetHandlerFunction_;
	PacketLogger packetLogger{"C:\\Users\\Victor\\Documents\\Development\\packet-logs\\"};

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
};

#endif