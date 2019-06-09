#include "proxy.hpp"

Proxy::Proxy(BrokerSystem &broker, uint16_t port) :
      broker_(broker),
      acceptor(ioService_, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
      timer(boost::make_shared<boost::asio::deadline_timer>(ioService_)) {
  broker_.setInjectionFunction(std::bind(&Proxy::inject, this, std::placeholders::_1, std::placeholders::_2));
  std::cout << "Proxy constructed, listening on port " << port << '\n';
  //Start accepting connections
  PostAccept();

  //Post the packet processing timer
  timer->expires_from_now(boost::posix_time::milliseconds(kPacketProcessDelayMs));
  timer->async_wait(boost::bind(&Proxy::ProcessPackets, this, boost::asio::placeholders::error));
}

Proxy::~Proxy() {
  //Stop everything (shutting down)
  Stop();
}

void Proxy::inject(const PacketContainer &packet, const PacketContainer::Direction direction) {
  if (direction == PacketContainer::Direction::kClientToServer) {
    if (serverConnection.security) {
      packetLogger.logPacket(packet, false, PacketLogger::Direction::BotToServer);
      serverConnection.Inject(packet);
    }
  } else if (direction == PacketContainer::Direction::kServerToClient) {
    if (clientConnection.security) {
      packetLogger.logPacket(packet, false, PacketLogger::Direction::BotToClient);
      clientConnection.Inject(packet);
    }
  }
}

void Proxy::start() {
  //Start processing network events
  std::cout << "Proxy starting\n";
  while(true) {
    try {
      //Run
      boost::system::error_code ec;
      ioService_.run(ec);

      if(ec) {
        std::cout << "[" << __FUNCTION__ << "]" << "[" << __LINE__ << "] " << ec.message() << std::endl;
      } else {
        //No more work
        break;
      }
      
      //Prevent high CPU usage
      boost::this_thread::sleep(boost::posix_time::milliseconds(1));
    }
    catch(std::exception & e) {
      std::cout << "[" << __FUNCTION__ << "]" << "[" << __LINE__ << "] " << e.what() << std::endl;
    }
  }
}

// Stops all networking objects
void Proxy::Stop() {
  boost::system::error_code ec;
  acceptor.close(ec);
  acceptor.cancel(ec);

  if(timer)
    timer->cancel(ec);

  clientConnection.Close();
  serverConnection.Close();
}

// Starts accepting new connections
void Proxy::PostAccept(uint32_t count) {
  for(uint32_t x = 0; x < count; ++x) {
    //The newly created socket will be used when something connects
    boost::shared_ptr<boost::asio::ip::tcp::socket> s(boost::make_shared<boost::asio::ip::tcp::socket>(ioService_));
    acceptor.async_accept(*s, boost::bind(&Proxy::HandleAccept, this, s, boost::asio::placeholders::error));
  }
}

// Handles new connections
void Proxy::HandleAccept(boost::shared_ptr<boost::asio::ip::tcp::socket> s, const boost::system::error_code & error) {
  std::cout << "Proxy received new connection\n";
  //Error check
  if(!error) {
    //Close active connections
    clientConnection.Close();
    serverConnection.Close();

    //Disable nagle
    s->set_option(boost::asio::ip::tcp::no_delay(true));

    clientConnection.Initialize(s);
    clientConnection.security->GenerateHandshake();

    boost::system::error_code ec;
    if (connectToAgent) {
      //Connect to the agent server
      std::cout << "Connecting to " << agentIP_ << ":" << agentPort_ << std::endl;
      ec = serverConnection.Connect(agentIP_, agentPort_);
    } else {
      //Connect to the gateway server
      std::cout << "Connecting to " << Config::GatewayIP << ":" << Config::GatewayPort << std::endl;
      ec = serverConnection.Connect(Config::GatewayIP, Config::GatewayPort);
    }

    //Error check
    if(ec) {
      std::cout << "[Error] Unable to connect to " << (connectToAgent ? agentIP_ : Config::GatewayIP) << ":" << (connectToAgent ? agentPort_ : Config::GatewayPort) << std::endl;
      std::cout << ec.message() << std::endl;

      //Silkroad connection is no longer needed
      clientConnection.Close();
    } else {
      clientConnection.PostRead();
      serverConnection.PostRead();
    }

    //Next connection goes to the gateway server
    connectToAgent = false;

    //Post another accept
    PostAccept();
  }
}

void Proxy::ProcessPackets(const boost::system::error_code & error) {
  if(!error) {
    if(clientConnection.security) {
      while(clientConnection.security->HasPacketToRecv()) {
        bool forward = true;

        //Retrieve the packet out of the security api
        PacketContainer p = clientConnection.security->GetPacketToRecv();

        //Check the blocked list
        if(blockedOpcodes_.find(p.opcode) != blockedOpcodes_.end()) {
          forward = false;
        }

        //Log packet
        packetLogger.logPacket(p, !forward, PacketLogger::Direction::kClientToServer);

        if(p.opcode == 0x2001) {
          std::cout << "\"Silkroad\" Connected" << std::endl;
          forward = false;
        }

        //Forward the packet to Joymax
        if(forward && serverConnection.security) {
          bool forwardToServer = broker_.packetReceived(p, PacketContainer::Direction::kClientToServer);
          if (forwardToServer) {
            serverConnection.Inject(p);
          }
        }
      }

      //Send packets that are currently in the security api
      while(clientConnection.security->HasPacketToSend()) {
        if(!clientConnection.Send(clientConnection.security->GetPacketToSend()))
          break;
      }
    }

    if(serverConnection.security) {
      while(serverConnection.security->HasPacketToRecv()) {
        bool forward = true;

        //Retrieve the packet out of the security api
        PacketContainer p = serverConnection.security->GetPacketToRecv();

        //Check the blocked list
        if(blockedOpcodes_.find(p.opcode) != blockedOpcodes_.end())
          forward = false;

        //Log packet
        packetLogger.logPacket(p, !forward, PacketLogger::Direction::kServerToClient);

        if(p.opcode == 0xA102) {
          StreamUtility r = p.data;
          if(r.Read<uint8_t>() == 1) {
              
            std::cout << "\"Joymax\" Connected" << std::endl;
            //Do not forward the packet to Joymax, we need to replace the agent server data
            forward = false;
            
            //The next connection will go to the agent server
            connectToAgent = true;

            uint32_t LoginID = r.Read<uint32_t>();				//Login ID
            agentIP_ = r.Read_Ascii(r.Read<uint16_t>());			//Agent IP
            agentPort_ = r.Read<uint16_t>();						//Agent port

            StreamUtility w;
            w.Write<uint8_t>(1);								//Success flag
            w.Write<uint32_t>(LoginID);							//Login ID
            w.Write<uint16_t>(9);								//Length of 127.0.0.1
            w.Write_Ascii("127.0.0.1");							//IP
            w.Write<uint16_t>(Config::BindPort);				//Port

            //Inject the packet
            clientConnection.Inject(p.opcode, w);

            //Inject the packet immediately
            while(clientConnection.security->HasPacketToSend())
              clientConnection.Send(clientConnection.security->GetPacketToSend());

            //Close active connections
            clientConnection.Close();
            serverConnection.Close();

            //Want to forward this to the bot so it can grab the token
            broker_.packetReceived(p, PacketContainer::Direction::kServerToClient);
            //Security pointer is now valid so skip to the end
            goto Post;
          }
        }

        //Forward the packet to Silkroad
        if(forward && clientConnection.security) {
          bool forwardToClient = broker_.packetReceived(p, PacketContainer::Direction::kServerToClient);
          if (forwardToClient) {
            clientConnection.Inject(p);
          }
        }
      }

      //Send packets that are currently in the security api
      while(serverConnection.security->HasPacketToSend()) {
        if(!serverConnection.Send(serverConnection.security->GetPacketToSend()))
          break;
      }
    }

Post:
    //Repost the timer
    timer->expires_from_now(boost::posix_time::milliseconds(kPacketProcessDelayMs));
    timer->async_wait(boost::bind(&Proxy::ProcessPackets, this, boost::asio::placeholders::error));
  }
}
