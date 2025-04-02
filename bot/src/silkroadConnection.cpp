#include "packet/opcode.hpp"
#include "silkroadConnection.hpp"

#include <tracy/Tracy.hpp>

#include <boost/bind/bind.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <thread>

#define CHECK_ERROR(error)                               \
do {                                                     \
  if (error) {                                           \
    LOG(ERROR) << "[" << name_ << "] Error: \"" << error.message() << '"'; \
  }                                                      \
} while (0)

//Handles incoming packets
void SilkroadConnection::HandleRead(size_t bytes_transferred, const boost::system::error_code & error) {
  if (!error && boostSocket_ && security) {
    security->Recv(&data[0], bytes_transferred);
    PostRead();
  } else if (!closingConnection_) {
    if (error == boost::asio::error::eof) {
      LOG(ERROR) << "[" << name_ << "] Connection closed by remote host";
      Close();
    } else {
      CHECK_ERROR(error);
    }
  }
}

SilkroadConnection::SilkroadConnection(boost::asio::io_service &ioService, std::string_view name) : ioService_(ioService), name_(name) {
  data.resize(kMaxPacketRecvSizeBytes + 1);
}

SilkroadConnection::~SilkroadConnection() {
  Close();
}

//Gets everything ready for receiving packets
void SilkroadConnection::Initialize(boost::shared_ptr<boost::asio::ip::tcp::socket> s_) {
  boostSocket_ = s_;
  security = boost::make_shared<SilkroadSecurity>();
}

//Starts receiving data
void SilkroadConnection::PostRead() {
  if (boostSocket_ && security) {
    // if (closingConnection_) {
    //   LOG(WARNING) << "Closing connection";
    // }
    boostSocket_->async_read_some(boost::asio::buffer(&data[0], kMaxPacketRecvSizeBytes), boost::bind(&SilkroadConnection::HandleRead, this, boost::asio::placeholders::bytes_transferred, boost::asio::placeholders::error));
  }
}

//Closes the socket
void SilkroadConnection::Close() {
  // if (closingConnection_) {
  //   // Already closing
  //   LOG(WARNING) << "Already closing connection";
  //   return;
  // }
  closingConnection_ = true;

  if (boostSocket_) {
    boost::system::error_code ec;
    boostSocket_->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    CHECK_ERROR(ec);
    boostSocket_->close(ec);
    CHECK_ERROR(ec);
    boostSocket_.reset();
  }

  security.reset();
}

boost::system::error_code SilkroadConnection::Connect(const std::string & IP, uint16_t port) {
  closingConnection_ = false;

  // Create the socket
  boostSocket_ = boost::make_shared<boost::asio::ip::tcp::socket>(ioService_);

  boost::system::error_code ec;
  boost::system::error_code resolve_ec;

  for(uint8_t x = 0; x < 3; ++x) {
    // Connect
    boostSocket_->connect(boost::asio::ip::tcp::endpoint(boost::asio::ip::address::from_string(IP, resolve_ec), port), ec);
    CHECK_ERROR(ec);

    // Probably not a valid IP so it's a hostname
    if (resolve_ec) {
      boost::asio::ip::tcp::resolver resolver(ioService_);
      boost::asio::ip::tcp::resolver::query query(boost::asio::ip::tcp::v4(), IP, boost::lexical_cast<std::string>(port));
      boost::asio::ip::tcp::resolver::iterator iterator = resolver.resolve(query);
      boostSocket_->connect(*iterator, ec);
      CHECK_ERROR(ec);
    }

    //See if there was an error
    if (!ec) break;

    //Error occurred so wait
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  if (!ec) {
    //Create new Silkroad security
    security = boost::make_shared<SilkroadSecurity>();

    // Disable nagle
    boostSocket_->set_option(boost::asio::ip::tcp::no_delay(true));
  }

  return ec;
}

// Insert packet into the outgoing packet list of the security API
bool SilkroadConnection::InjectToSend(uint16_t opcode, StreamUtility & p, bool encrypted) {
  if (security) {
    security->Send(opcode, p, encrypted ? 1 : 0, 0);
    return true;
  }

  return false;
}

// Insert packet into the outgoing packet list of the security API
bool SilkroadConnection::InjectToSend(uint16_t opcode, bool encrypted) {
  if (security) {
    security->Send(opcode, 0, 0, encrypted ? 1 : 0, 0);
    return true;
  }

  return false;
}

// Insert packet into the outgoing packet list of the security API
bool SilkroadConnection::InjectToSend(const PacketContainer &container) {
  if (security) {
    security->Send(container.opcode, container.data, container.encrypted, container.massive);
    return true;
  }

  return false;
}

// Insert packet into the incoming packet list of the security API
bool SilkroadConnection::InjectAsReceived(const PacketContainer &container) {
  if (security) {
#if TRACY_ENABLE
    {
      const std::string messageWithOpcode = absl::StrFormat("SilkroadConnection::InjectAsReceived::%s", packet::toString(static_cast<packet::Opcode>(container.opcode)));
      TracyMessage(messageWithOpcode.c_str(), messageWithOpcode.size());
    }
#endif
    security->Recv(container);
    return true;
  }

  return false;
}

//Sends a formatted packet
bool SilkroadConnection::Send(const std::vector<uint8_t> &packet) {
  if (!boostSocket_) {
    LOG(WARNING) << "Trying to send packet without socket";
    return false;
  }

  // Send the packet all at once
  boost::system::error_code ec;
  boost::asio::write(*boostSocket_, boost::asio::buffer(&packet[0], packet.size()), boost::asio::transfer_all(), ec);

  // See if there was an error
  if (ec) {
    LOG(ERROR) << "Error sending packet: " << ec.message();
    Close();
    return false;
  }

  return true;
}
