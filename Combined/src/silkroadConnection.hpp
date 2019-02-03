#include "shared/silkroad_security.h"

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <boost/asio.hpp>

#ifndef SILKROAD_CONNECTION_HPP_
#define SILKROAD_CONNECTION_HPP_

namespace Config {
	//Gateway server info
	extern std::string GatewayIP;	//Gateway server IP/hostname to connect to
	extern uint16_t GatewayPort;											//Gateway server port
	//Listen info
	extern uint16_t BindPort;													//Gateway server bind port
	extern uint16_t BotBind;													//The port the bot will connect to
	//Data
	extern uint32_t DataMaxSize;											//The maximum number of bytes to receive in one packet
};

//Silkroad connection class
class SilkroadConnection {
private:
  boost::asio::io_service &ioService_;

	//Socket
	boost::shared_ptr<boost::asio::ip::tcp::socket> s;

	//Data
	std::vector<uint8_t> data;

	//Handles incoming packets
	void HandleRead(size_t bytes_transferred, const boost::system::error_code & error);

public:

	//Security
	boost::shared_ptr<SilkroadSecurity> security;

	//Constructor
	SilkroadConnection(boost::asio::io_service &ioService);

	//Destructor
	~SilkroadConnection();

	//Gets everything ready for receiving packets
	void Initialize(boost::shared_ptr<boost::asio::ip::tcp::socket> s_);

	//Starts receiving data
	void PostRead();

	//Closes the socket
	void Close();

	boost::system::error_code Connect(const std::string & IP, uint16_t port);

	//Hands packets off to the security API
	bool Inject(uint16_t opcode, StreamUtility & p, bool encrypted = false);

	//Hands packets off to the security API
	bool Inject(uint16_t opcode, bool encrypted = false);

	//Hands packets off to the security API
	bool Inject(PacketContainer & container);

	//Sends a formatted packet
	bool Send(const std::vector<uint8_t> & packet);
};

#endif