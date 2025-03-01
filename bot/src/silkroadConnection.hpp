#include "shared/silkroad_security.h"

#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <boost/asio.hpp>

#ifndef SILKROAD_CONNECTION_HPP_
#define SILKROAD_CONNECTION_HPP_

//Silkroad connection class
class SilkroadConnection {
private:
  static const uint32_t kMaxPacketRecvSizeBytes{16384}; //The maximum number of bytes to receive in one packet
  boost::asio::io_service &ioService_;

  bool closingConnection_{false};

	//Socket
	boost::shared_ptr<boost::asio::ip::tcp::socket> boostSocket_;

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

	// Insert packet into the outgoing packet list of the security API
	bool InjectToSend(uint16_t opcode, StreamUtility & p, bool encrypted = false);

	// Insert packet into the outgoing packet list of the security API
	bool InjectToSend(uint16_t opcode, bool encrypted = false);

	// Insert packet into the outgoing packet list of the security API
	bool InjectToSend(const PacketContainer &container);

	// Insert packet into the incoming packet list of the security API
	bool InjectAsReceived(const PacketContainer &container);

	//Sends a formatted packet
	bool Send(const std::vector<uint8_t> & packet);
};

#endif