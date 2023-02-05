#include "logging.hpp"
#include "silkroadConnection.hpp"

#define CHECK_ERROR(error) \
if (error) { \
  LOG() << "Error: \"" << error.message() << '"' << std::endl; \
}

//Handles incoming packets
void SilkroadConnection::HandleRead(size_t bytes_transferred, const boost::system::error_code & error) {
  if (!error && s && security) {
    security->Recv(&data[0], bytes_transferred);
    PostRead();
  } else if (!closingConnection_) {
    CHECK_ERROR(error);
  }
}

//Constructor
SilkroadConnection::SilkroadConnection(boost::asio::io_service &ioService) : ioService_(ioService) {
  data.resize(kMaxPacketRecvSizeBytes + 1);
}

//Destructor
SilkroadConnection::~SilkroadConnection() {
  Close();
}

//Gets everything ready for receiving packets
void SilkroadConnection::Initialize(boost::shared_ptr<boost::asio::ip::tcp::socket> s_) {
  s = s_;
  security = boost::make_shared<SilkroadSecurity>();
}

//Starts receiving data
void SilkroadConnection::PostRead() {
  if(s && security) {
    s->async_read_some(boost::asio::buffer(&data[0], kMaxPacketRecvSizeBytes), boost::bind(&SilkroadConnection::HandleRead, this, boost::asio::placeholders::bytes_transferred, boost::asio::placeholders::error));
  }
}

//Closes the socket
void SilkroadConnection::Close() {
  closingConnection_ = true;

  if (s) {
    boost::system::error_code ec;
    s->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    CHECK_ERROR(ec);
    s->close(ec);
    CHECK_ERROR(ec);
    s.reset();
  }

  security.reset();
}

boost::system::error_code SilkroadConnection::Connect(const std::string & IP, uint16_t port) {
  closingConnection_ = false;

  //Create the socket
  s = boost::make_shared<boost::asio::ip::tcp::socket>(ioService_);

  boost::system::error_code ec;
  boost::system::error_code resolve_ec;

  for(uint8_t x = 0; x < 3; ++x) {
    //Connect
    s->connect(boost::asio::ip::tcp::endpoint(boost::asio::ip::address::from_string(IP, resolve_ec), port), ec);
    CHECK_ERROR(ec);

    //Probably not a valid IP so it's a hostname
    if (resolve_ec) {
      boost::asio::ip::tcp::resolver resolver(ioService_);
      boost::asio::ip::tcp::resolver::query query(boost::asio::ip::tcp::v4(), IP, boost::lexical_cast<std::string>(port));
      boost::asio::ip::tcp::resolver::iterator iterator = resolver.resolve(query);
      s->connect(*iterator, ec);
      CHECK_ERROR(ec);
    }

    //See if there was an error
    if (!ec) break;

    //Error occurred so wait
    boost::this_thread::sleep(boost::posix_time::milliseconds(500));
  }

  if (!ec) {
    //Create new Silkroad security
    security = boost::make_shared<SilkroadSecurity>();

    //Disable nagle
    s->set_option(boost::asio::ip::tcp::no_delay(true));
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
    security->Recv(container);
    return true;
  }

  return false;
}

//Sends a formatted packet
bool SilkroadConnection::Send(const std::vector<uint8_t> & packet) {
  if(!s) return false;

  //Send the packet all at once
  boost::system::error_code ec;
  boost::asio::write(*s, boost::asio::buffer(&packet[0], packet.size()), boost::asio::transfer_all(), ec);
  CHECK_ERROR(ec);
  
  //See if there was an error
  if (ec) {
    Close();
    return false;
  }

  return true;
}
