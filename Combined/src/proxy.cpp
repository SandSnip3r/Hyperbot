#include "logging.hpp"
#include "proxy.hpp"

Proxy::Proxy(const pk2::GameData &gameData, broker::PacketBroker &broker, uint16_t port) :
      gatewayPort_(gameData.gatewayPort()),
      divisionInfo_(gameData.divisionInfo()),
      packetBroker_(broker),
      acceptor(ioService_, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
      timer(boost::make_shared<boost::asio::deadline_timer>(ioService_)) {
  // Get the port that the OS gave us to listen on
  if (port == 0) {
    // Using an OS-assigned port
    boost::asio::ip::tcp::endpoint localEndpoint = acceptor.local_endpoint();
    ourListeningPort_ = localEndpoint.port();
  } else {
    // Using the given port
    ourListeningPort_ = port;
  }

  if (divisionInfo_.divisions.empty()) {
    throw std::runtime_error("Proxy given DivisionInfo without any divisions");
  }
  if (divisionInfo_.divisions[0].gatewayIpAddresses.empty()) {
    throw std::runtime_error("Proxy given Division without any addresses");
  }
  gatewayAddress_ = divisionInfo_.divisions[0].gatewayIpAddresses[0];

  packetBroker_.setInjectionFunction(std::bind(&Proxy::inject, this, std::placeholders::_1, std::placeholders::_2));
  std::cout << "Proxy constructed, listening on port " << ourListeningPort_ << '\n';
  //Start accepting connections
  PostAccept();

  //Post the packet processing timer
  timer->expires_from_now(boost::posix_time::milliseconds(kPacketProcessDelayMs));
  timer->async_wait(boost::bind(&Proxy::ProcessPackets, this, boost::asio::placeholders::error));
}

Proxy::~Proxy() {
  // Stop everything (shutting down)
  stop();
}

void Proxy::inject(const PacketContainer &packet, const PacketContainer::Direction direction) {
  // Inject into the incoming stream of the source, rather than the outgoing stream for the destination.
  //  This allows us to run injected packets through the same pipeline as normal packets, i.e. we can subscribe to our own injected packets.
  if (direction == PacketContainer::Direction::kClientToServer ||
      direction == PacketContainer::Direction::kBotToServer) {
    if (clientConnection.security) {
      clientConnection.InjectAsReceived(packet);
    }
  } else if (direction == PacketContainer::Direction::kServerToClient ||
             direction == PacketContainer::Direction::kBotToClient) {
    if (serverConnection.security) {
      serverConnection.InjectAsReceived(packet);
    }
  } else {
    throw std::runtime_error("Packet inject with unhandled direction: " + std::to_string(static_cast<int>(direction)));
  }
}

void Proxy::run() {
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

uint16_t Proxy::getOurListeningPort() const {
  return ourListeningPort_;
}

// Stops all networking objects
void Proxy::stop() {
  boost::system::error_code ec;
  acceptor.close(ec);
  acceptor.cancel(ec);

  if (timer) {
    timer->cancel(ec);
  }

  clientConnection.Close();
  serverConnection.Close();
}

void Proxy::blockOpcode(packet::Opcode opcode) {
  blockedOpcodes_.emplace(static_cast<std::underlying_type_t<packet::Opcode>>(opcode));
}

void Proxy::unblockOpcode(packet::Opcode opcode) {
  if (auto it = blockedOpcodes_.find(static_cast<std::underlying_type_t<packet::Opcode>>(opcode)); it != blockedOpcodes_.end()) {
    blockedOpcodes_.erase(it);
  }
}

bool Proxy::blockingOpcode(packet::Opcode opcode) const {
  return (blockedOpcodes_.find(static_cast<std::underlying_type_t<packet::Opcode>>(opcode)) != blockedOpcodes_.end());
}

// Starts accepting new connections
void Proxy::PostAccept(uint32_t count) {
  for (uint32_t x = 0; x < count; ++x) {
    //The newly created socket will be used when something connects
    boost::shared_ptr<boost::asio::ip::tcp::socket> s(boost::make_shared<boost::asio::ip::tcp::socket>(ioService_));
    acceptor.async_accept(*s, boost::bind(&Proxy::HandleAccept, this, s, boost::asio::placeholders::error));
  }
}

// Handles new connections
void Proxy::HandleAccept(boost::shared_ptr<boost::asio::ip::tcp::socket> s, const boost::system::error_code & error) {
  std::cout << "Proxy received new connection\n";
  //Error check
  if (!error) {
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
      std::cout << "Connecting to agent server at " << agentIP_ << ":" << agentPort_ << std::endl;
      ec = serverConnection.Connect(agentIP_, agentPort_);
    } else {
      //Connect to the gateway server
      std::cout << "Connecting to gateway server at " << gatewayAddress_ << ":" << gatewayPort_ << std::endl;
      ec = serverConnection.Connect(gatewayAddress_, gatewayPort_);
    }

    //Error check
    if(ec) {
      std::cout << "[Error] Unable to connect to " << (connectToAgent ? agentIP_ : gatewayAddress_) << ":" << (connectToAgent ? agentPort_ : gatewayPort_) << std::endl;
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
  } else {
    std::cout << "Proxy::HandleAccept error" << std::endl;
  }
}

void Proxy::ProcessPackets(const boost::system::error_code & error) {
  if (!error) {
    if (clientConnection.security) {
      while (clientConnection.security->HasPacketToRecv()) {
        // Client sent a packet
        bool forward = true;

        //Retrieve the packet out of the security api
        const auto [packet, wasInjected] = clientConnection.security->GetPacketToRecv();
        const auto direction = (wasInjected ? PacketContainer::Direction::kBotToServer : PacketContainer::Direction::kClientToServer);

        // Check the blocked list
        // Don't block injected packets
        if (!wasInjected && blockedOpcodes_.find(packet.opcode) != blockedOpcodes_.end()) {
          forward = false;
        }

        //Log packet
        packetLogger.logPacket(packet, !forward, direction);

        if (packet.opcode == 0x2001) {
          forward = false;
        }

        // Run packet through bot, regardless if it's blocked
        packetBroker_.packetReceived(packet, direction);

        //Forward the packet to Joymax
        if (forward && serverConnection.security) {
          serverConnection.InjectToSend(packet);
        }
      }

      //Send packets that are currently in the security api
      while (clientConnection.security->HasPacketToSend()) {
        if (!clientConnection.Send(clientConnection.security->GetPacketToSend())) {
          LOG() << "Client connection Send error" << std::endl;
          break;
        }
      }
    }

    if (serverConnection.security) {
      while (serverConnection.security->HasPacketToRecv()) {
        // Server sent a packet
        bool forward = true;

        //Retrieve the packet out of the security api
        const auto [packet, wasInjected] = serverConnection.security->GetPacketToRecv();
        const auto direction = (wasInjected ? PacketContainer::Direction::kBotToClient : PacketContainer::Direction::kServerToClient);

        // Check the blocked list
        // Don't block injected packets
        if (!wasInjected && blockedOpcodes_.find(packet.opcode) != blockedOpcodes_.end()) {
          forward = false;
        }

        //Log packet
        packetLogger.logPacket(packet, !forward, direction);

        if (static_cast<packet::Opcode>(packet.opcode) == packet::Opcode::kServerGatewayLoginResponse) {
          StreamUtility r = packet.data;
          if (r.Read<uint8_t>() == 1) {
            //The next connection will go to the agent server
            connectToAgent = true;

            uint32_t loginID = r.Read<uint32_t>();        //Login ID
            agentIP_ = r.Read_Ascii(r.Read<uint16_t>());  //Agent IP
            agentPort_ = r.Read<uint16_t>();              //Agent port

            StreamUtility w;
            w.Write<uint8_t>(1);                    // Success flag
            w.Write<uint32_t>(loginID);             // Login ID
            w.Write<uint16_t>(9);                   // Length of 127.0.0.1
            w.Write_Ascii("127.0.0.1");             // IP
            w.Write<uint16_t>(ourListeningPort_);   // Port

            //Inject the packet
            clientConnection.InjectToSend(packet.opcode, w);

            // Inject the packet immediately
            while (clientConnection.security->HasPacketToSend()) {
              clientConnection.Send(clientConnection.security->GetPacketToSend());
            }

            //Close active connections
            LOG() << "Closing gateway connection, connecting to agentserver" << std::endl;
            clientConnection.Close();
            serverConnection.Close();

            //Want to forward this to the bot so it can grab the token
            packetBroker_.packetReceived(packet, direction);
            //Security pointer is now valid so skip to the end
            // Skipping to "Post" wont forward this packet to the client
            goto Post;
          }
        }

        const auto opcodeAsEnum = static_cast<packet::Opcode>(packet.opcode);
        if (opcodeAsEnum == packet::Opcode::SERVER_AGENT_CHARACTER_INFO_BEGIN) {
          // Initialize data/container
          if (characterInfoPacketContainer_) {
            // What? There's already one?
            std::cout << "[@@@] Wait, we got a char info begin packet, but we've already initialized the data\n";
          }
          // Initialize packet data with the "begin" data
          characterInfoPacketContainer_.emplace(packet);
          // Update opcode to reflect "data"
          characterInfoPacketContainer_->opcode = static_cast<uint16_t>(packet::Opcode::kServerAgentCharacterData);
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentEntityGroupspawnBegin) {
          // Initialize data/container
          if (groupSpawnPacketContainer_) {
            // What? There's already one?
            std::cout << "[@@@] Wait, we got a char info begin packet, but we've already initialized the data\n";
          }
          // Initialize packet data with the "begin" data
          groupSpawnPacketContainer_.emplace(packet);
          // Update opcode to reflect "data"
          groupSpawnPacketContainer_->opcode = static_cast<uint16_t>(packet::Opcode::kServerAgentEntityGroupspawnData);
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentInventoryStorageBegin) {
          // Initialize data/container
          if (storagePacketContainer_) {
            // What? There's already one?
            std::cout << "[@@@] Wait, we got a storage begin packet, but we've already initialized the data\n";
          }
          // Initialize packet data with the "begin" data
          storagePacketContainer_.emplace(packet);
          // Update opcode to reflect "data"
          storagePacketContainer_->opcode = static_cast<uint16_t>(packet::Opcode::kServerAgentInventoryStorageData);
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentGuildStorageBegin) {
          // Initialize data/container
          if (guildStoragePacketContainer_) {
            // What? There's already one?
            std::cout << "[@@@] Wait, we got a guild storage begin packet, but we've already initialized the data\n";
          }
          // Initialize packet data with the "begin" data
          guildStoragePacketContainer_.emplace(packet);
          // Update opcode to reflect "data"
          guildStoragePacketContainer_->opcode = static_cast<uint16_t>(packet::Opcode::kServerAgentGuildStorageData);
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentCharacterData) {
          // Append all data to container
          characterInfoPacketContainer_->data.Write(packet.data.GetStreamVector());
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentEntityGroupspawnData) {
          // Append all data to container
          groupSpawnPacketContainer_->data.Write(packet.data.GetStreamVector());
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentInventoryStorageData) {
          // Append all data to container
          storagePacketContainer_->data.Write(packet.data.GetStreamVector());
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentGuildStorageData) {
          // Append all data to container
          guildStoragePacketContainer_->data.Write(packet.data.GetStreamVector());
        }

        // Run packet through bot, regardless if it's blocked
        // Handle "begin", "data", and "end" pieces of split packets
        if (opcodeAsEnum == packet::Opcode::SERVER_AGENT_CHARACTER_INFO_END) {
          // Send packet to broker
          packetBroker_.packetReceived(*characterInfoPacketContainer_, direction);
          // Reset data
          characterInfoPacketContainer_.reset();
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentEntityGroupspawnEnd) {
          // Send packet to broker
          packetBroker_.packetReceived(*groupSpawnPacketContainer_, direction);
          // Reset data
          groupSpawnPacketContainer_.reset();
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentInventoryStorageEnd) {
          // Send packet to broker
          packetBroker_.packetReceived(*storagePacketContainer_, direction);
          // Reset data
          storagePacketContainer_.reset();
        } else if (opcodeAsEnum == packet::Opcode::kServerAgentGuildStorageEnd) {
          // Send packet to broker
          packetBroker_.packetReceived(*guildStoragePacketContainer_, direction);
          // Reset data
          guildStoragePacketContainer_.reset();
        } else if (opcodeAsEnum != packet::Opcode::SERVER_AGENT_CHARACTER_INFO_BEGIN &&
                    opcodeAsEnum != packet::Opcode::kServerAgentCharacterData &&
                    opcodeAsEnum != packet::Opcode::kServerAgentEntityGroupspawnBegin &&
                    opcodeAsEnum != packet::Opcode::kServerAgentEntityGroupspawnData &&
                    opcodeAsEnum != packet::Opcode::kServerAgentInventoryStorageBegin &&
                    opcodeAsEnum != packet::Opcode::kServerAgentInventoryStorageData &&
                    opcodeAsEnum != packet::Opcode::kServerAgentGuildStorageBegin &&
                    opcodeAsEnum != packet::Opcode::kServerAgentGuildStorageData) {
          // In all other cases, if its not "begin" or "data", send it
          packetBroker_.packetReceived(packet, direction);
        }

        //Forward the packet to the Silkroad server
        if (forward && clientConnection.security) {
          clientConnection.InjectToSend(packet);
        }
      }

      //Send packets that are currently in the security api
      while (serverConnection.security->HasPacketToSend()) {
        if (!serverConnection.Send(serverConnection.security->GetPacketToSend())) {
          LOG() << "Server connection Send error" << std::endl;
          break;
        }
      }
    }

Post:
    //Repost the timer
    timer->expires_from_now(boost::posix_time::milliseconds(kPacketProcessDelayMs));
    timer->async_wait(boost::bind(&Proxy::ProcessPackets, this, boost::asio::placeholders::error));
  }
}
