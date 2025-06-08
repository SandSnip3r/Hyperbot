#include "proxy.hpp"
#include "packet/building/clientGatewayPatchRequest.hpp"
#include "packet/building/frameworkAliveNotify.hpp"
#include "packet/building/serverGatewayLoginResponse.hpp"
#include "packet/parsing/serverGatewayLoginResponse.hpp"

#include <common/TracySystem.hpp>
#include <tracy/Tracy.hpp>

#include <boost/bind/bind.hpp>

#include <absl/log/log.h>

#include <thread>

Proxy::Proxy(const pk2::GameData &gameData, broker::PacketBroker &broker, uint16_t port) :
      gatewayPort_(gameData.gatewayPort()),
      divisionInfo_(gameData.divisionInfo()),
      packetBroker_(broker),
      acceptor(ioService_, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
      packetProcessingTimer_(boost::make_shared<boost::asio::deadline_timer>(ioService_)) {
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
  VLOG(1) << "Constructed Proxy with listening port " << ourListeningPort_ << " and gateway address " << gatewayAddress_;

  packetBroker_.setInjectionFunction(std::bind(&Proxy::inject, this, std::placeholders::_1, std::placeholders::_2));

  // Start accepting connections
  PostAccept();

  // Post the packet processing timer
  packetProcessingTimer_->expires_from_now(boost::posix_time::milliseconds(kPacketProcessDelayMs));
  packetProcessingTimer_->async_wait(boost::bind(&Proxy::ProcessPackets, this, boost::asio::placeholders::error));
}

Proxy::~Proxy() {
  // Stop everything (shutting down)
  stop();
  if (thr_.joinable()) {
    thr_.join();
  }
  VLOG(1) << "Destructing";
}

void Proxy::inject(const PacketContainer &packet, const PacketContainer::Direction direction) {
  // Inject into the incoming stream of the source, rather than the outgoing stream for the destination.
  //  This allows us to run injected packets through the same pipeline as normal packets, i.e. we can subscribe to our own injected packets.
  if (direction == PacketContainer::Direction::kClientToServer ||
      direction == PacketContainer::Direction::kBotToServer) {
    if (clientless_) {
      injectedClientPacketsForClientless_.push_back(packet);
    } else {
      if (clientConnection.security) {
        clientConnection.InjectAsReceived(packet);
      }
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

void Proxy::runAsync() {
  if (thr_.joinable()) {
    throw std::runtime_error("Proxy::runAsync called while already running");
  }
  thr_ = std::thread(&Proxy::run, this);
}

void Proxy::run() {
  tracy::SetThreadName("Proxy");
  // Start processing network events
  while(true) {
    try {
      // Run
      boost::system::error_code ec;
      ioService_.run(ec);

      if (ec) {
        LOG(INFO) << "Error running io_service: \"" << ec.message() << '"';
      } else {
        LOG(INFO) << "No more work";
        // No more work
        break;
      }

      // Prevent high CPU usage
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } catch (const std::exception &ex) {
      LOG(ERROR) << "Exception while running io_service \"" << ex.what() << '"';
    }
  }
  LOG(INFO) << "Proxy::run exiting";
}

uint16_t Proxy::getOurListeningPort() const {
  return ourListeningPort_;
}

// Stops all networking objects
void Proxy::stop() {
  boost::system::error_code ec;
  acceptor.close(ec);
  if (ec) {
    LOG(ERROR) << "Error closing acceptor: " << ec.message();
  }
  acceptor.cancel(ec);
  if (ec) {
    LOG(ERROR) << "Error cancelling acceptor: " << ec.message();
  }

  if (packetProcessingTimer_) {
    packetProcessingTimer_->cancel(ec);
    if (ec) {
      LOG(ERROR) << "Error cancelling packet processing timer: " << ec.message();
    }
  }
  if (keepAlivePacketTimer_) {
    keepAlivePacketTimer_->cancel(ec);
    if (ec) {
      LOG(ERROR) << "Error cancelling keepalive timer: " << ec.message();
    }
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

void Proxy::setClientless(bool clientless) {
  // For now, we only support switching from clientful to clientless.
  if (!clientless_ && clientless) {
    VLOG(1) << "Proxy is switching to clientless";
    clientless_ = clientless;
    // Construct a timer to send a keepalive based on when we last sent a packet to the server.
    keepAlivePacketTimer_ = boost::make_shared<boost::asio::deadline_timer>(ioService_);
    setKeepaliveTimer();
  }
}

// Starts accepting new connections
void Proxy::PostAccept() {
  // The newly created socket will be used when something connects
  boost::shared_ptr<boost::asio::ip::tcp::socket> s(boost::make_shared<boost::asio::ip::tcp::socket>(ioService_));
  acceptor.async_accept(*s, boost::bind(&Proxy::HandleAccept, this, s, boost::asio::placeholders::error));
}

// Handles new connections
void Proxy::HandleAccept(boost::shared_ptr<boost::asio::ip::tcp::socket> s, const boost::system::error_code & error) {
  if (error) {
    LOG(WARNING) << "Error accepting new connection: \"" << error.message() << '"';
    return;
  }
  VLOG(1) << "New connection accepted";

  // Close active connections
  clientConnection.Close();
  serverConnection.Close();

  // Disable nagle
  s->set_option(boost::asio::ip::tcp::no_delay(true));
  s->set_option(boost::asio::socket_base::keep_alive(true));

  clientConnection.Initialize(s);
  clientConnection.security->GenerateHandshake();

  boost::system::error_code ec;
  if (connectToAgent_) {
    // Connect to the agent server
    VLOG(1) << absl::StreamFormat("Received connection; connecting to the agent server at %s:%d", agentIP_, agentPort_);
    ec = serverConnection.Connect(agentIP_, agentPort_);
  } else {
    // Connect to the gateway server
    VLOG(1) << absl::StreamFormat("Received connection; connecting to the gateway server at %s:%d", gatewayAddress_, gatewayPort_);
    ec = serverConnection.Connect(gatewayAddress_, gatewayPort_);
  }

  // Error check
  if (ec) {
    LOG(INFO) << "Unable to connect to " << (connectToAgent_ ? agentIP_ : gatewayAddress_) << ":" << (connectToAgent_ ? agentPort_ : gatewayPort_) << ": \"" << ec.message() << '"';

    // Silkroad connection is no longer needed
    clientConnection.Close();
  } else {
    clientConnection.PostRead();
    serverConnection.PostRead();
  }

  // Next connection goes to the gateway server
  connectToAgent_ = false;

  // Post another accept
  PostAccept();
}

void Proxy::connectClientlessAsync() {
  clientless_ = true;
  ioService_.post(boost::bind(&Proxy::connectClientless, this));

  // Construct a timer to send a keepalive based on when we last sent a packet to the server.
  keepAlivePacketTimer_ = boost::make_shared<boost::asio::deadline_timer>(ioService_);
  setKeepaliveTimer();
}

void Proxy::connectClientless() {
  boost::system::error_code ec;
  if (connectToAgent_) {
    // Connect to the agent server
    LOG(INFO) << absl::StreamFormat("Connecting clientlessly to the agent server at %s:%d", agentIP_, agentPort_);
    ec = serverConnection.Connect(agentIP_, agentPort_);
  } else {
    // Connect to the gateway server
    static std::atomic<int> count=0;
    LOG(INFO) << absl::StreamFormat("#%d Connecting clientlessly to the gateway server at %s:%d", count++, gatewayAddress_, gatewayPort_);
    ec = serverConnection.Connect(gatewayAddress_, gatewayPort_);
  }

  // Error check
  if (ec) {
    LOG(INFO) << "Unable to connect to " << (connectToAgent_ ? agentIP_ : gatewayAddress_) << ":" << (connectToAgent_ ? agentPort_ : gatewayPort_) << ": \"" << ec.message() << '"';
  } else {
    serverConnection.PostRead();
  }

  // Next connection goes to the gateway server
  // connectToAgent_ = false;
}

void Proxy::ProcessPackets(const boost::system::error_code &error) {
  ZoneScopedN("Proxy::ProcessPackets");
  if (error) {
    LOG(ERROR) << "Error during async_wait for ProcessPackets:\"" << error.message() << '"';
    // Not reposting timer.
    return;
  }

  try {
    receivePacketsFromClient();
    sendPacketsToClient();
    receivePacketsFromServer();
    sendPacketsToServer();
  } catch (const std::exception &ex) {
    LOG(ERROR) << "Exception while processing packets: \"" << ex.what() << '"';
  }

  // Repost the timer
  packetProcessingTimer_->expires_from_now(boost::posix_time::milliseconds(kPacketProcessDelayMs));
  packetProcessingTimer_->async_wait(boost::bind(&Proxy::ProcessPackets, this, boost::asio::placeholders::error));
}

void Proxy::setKeepaliveTimer() {
  // Set the timer to trigger when we need to send a keepalive packet.
  std::chrono::duration timeSinceLastPacketToServer = std::chrono::steady_clock::now() - lastPacketSentToServer_;
  const int millisecondsUntilNextKeepalive = std::floor(kMillisecondsBetweenKeepalives - timeSinceLastPacketToServer.count()/1'000'000.0);
  keepAlivePacketTimer_->expires_from_now(boost::posix_time::milliseconds(millisecondsUntilNextKeepalive));
  keepAlivePacketTimer_->async_wait(boost::bind(&Proxy::checkClientlessKeepalive, this, boost::asio::placeholders::error));
}

void Proxy::checkClientlessKeepalive(const boost::system::error_code &error) {
  // We've been woken up because we might need to send a keepalive packet.
  // How long has it been since we've last sent a packet?
  const std::chrono::duration timeSinceLastPacketToServer = std::chrono::steady_clock::now() - lastPacketSentToServer_;
  int millisecondsUntilNextKeepalive = std::floor(kMillisecondsBetweenKeepalives - timeSinceLastPacketToServer.count()/1'000'000.0);
  if (millisecondsUntilNextKeepalive <= 0) {
    // It has been long enough and we now need to send a keepalive.
    const PacketContainer packet = packet::building::FrameworkAliveNotify::packet();

    packetLogger.logPacket(packet, /*blocked=*/false, PacketContainer::Direction::kBotToServer);

    // Inject the packet immediately
    serverConnection.InjectToSend(packet);
    while (serverConnection.security->HasPacketToSend()) {
      serverConnection.Send(serverConnection.security->GetPacketToSend());
    }

    lastPacketSentToServer_ = std::chrono::steady_clock::now();
  }
  setKeepaliveTimer();
}


void Proxy::receivePacketsFromClient() {
  if (clientless_) {
    // In clientless mode, rather than pulling injected packets from the security API, we have a separate queue for injected packets.
    while (!injectedClientPacketsForClientless_.empty()) {
      const PacketContainer packet = injectedClientPacketsForClientless_.front();
      injectedClientPacketsForClientless_.pop_front();

      // Injected packets are always from the bot to the server.
      const PacketContainer::Direction direction = PacketContainer::Direction::kBotToServer;

      // Log packet
      packetLogger.logPacket(packet, /*blocked=*/false, direction);

      // Run packet through bot, regardless if it's blocked
      packetBroker_.packetReceived(packet, direction);

      // Forward the packet to gateway/agent server.
      if (serverConnection.security) {
        serverConnection.InjectToSend(packet);
      }
    }
  } else {
    if (!clientConnection.security) {
      // No client connection, no packets to receive
      return;
    }
    // Receive all pending incoming packets from the client
    while (clientConnection.security->HasPacketToRecv()) {
      // Client sent a packet
      bool forward = true;

      // Retrieve the packet out of the security api
      const auto [packet, wasInjected] = clientConnection.security->GetPacketToRecv();
      const PacketContainer::Direction direction = (wasInjected ? PacketContainer::Direction::kBotToServer : PacketContainer::Direction::kClientToServer);

      // Check the blocked list, but don't block injected packets
      if (!wasInjected && blockedOpcodes_.find(packet.opcode) != blockedOpcodes_.end()) {
        forward = false;
      }

      if (static_cast<packet::Opcode>(packet.opcode) == packet::Opcode::kFrameworkMessageIdentify) {
        // For some reason, we do not forward these messages to the server
        forward = false;
      }

      // Log packet
      packetLogger.logPacket(packet, !forward, direction);

      // Run packet through bot, regardless if it's blocked
      packetBroker_.packetReceived(packet, direction);

      // Forward the packet to gateway/agent server.
      if (forward && serverConnection.security) {
        serverConnection.InjectToSend(packet);
      }
    }
  }
}

void Proxy::sendPacketsToClient() {
  if (clientless_) {
    // No client, nothing to send.
    return;
  }
  if (!clientConnection.security) {
    // No client connection, nobody to send packets to
    return;
  }
  // Send all pending outgoing packets to the client
  while (clientConnection.security->HasPacketToSend()) {
    if (!clientConnection.Send(clientConnection.security->GetPacketToSend())) {
      LOG(ERROR) << "Client connection Send error. Closing connection.";
      clientConnection.Close();
      LOG(INFO) << "Going clientless";
      setClientless(true);
      break;
    }
  }
}

void Proxy::receivePacketsFromServer() {
  if (!serverConnection.security) {
    // No server connection, nothing to receive
    return;
  }
  while (serverConnection.security->HasPacketToRecv()) {
    // We received a packet from the server, handle it
    bool forward = true;

    // Retrieve the packet out of the security api
    const auto [packet, wasInjected] = serverConnection.security->GetPacketToRecv();
    const auto direction = (wasInjected ? PacketContainer::Direction::kBotToClient : PacketContainer::Direction::kServerToClient);

    // Check the blocked list, but don't block injected packets
    if (!wasInjected && blockedOpcodes_.find(packet.opcode) != blockedOpcodes_.end()) {
      forward = false;
    }

    // Log packet
    packetLogger.logPacket(packet, !forward, direction);

    // Do some special handling for packets which affect our connection state
    if (static_cast<packet::Opcode>(packet.opcode) == packet::Opcode::kServerGatewayLoginResponse) {
      packet::parsing::ServerGatewayLoginResponse serverGatewayLoginResponsePacket(packet);
      if (serverGatewayLoginResponsePacket.result() == packet::enums::LoginResult::kSuccess) {
        // Successful login
        // Save address of the actual agent server to later use
        agentIP_ = serverGatewayLoginResponsePacket.agentServerIp();
        agentPort_ = serverGatewayLoginResponsePacket.agentServerPort();

        // The next connection will go to the agent server
        connectToAgent_ = true;

        if (!clientless_) {
          if (!clientConnection.security) {
            throw std::runtime_error("Received gateway login response and not in clientless mode, but also have no client connection");
          }
          // Rewrite the login response packet which goes to the client with the bot's address and port
          const PacketContainer rewrittenLoginResponsePacket = packet::building::ServerGatewayLoginResponse::success(
              serverGatewayLoginResponsePacket.agentServerToken(),
              "127.0.0.1",
              ourListeningPort_
          );

          // Inject the packet immediately
          const bool injectSuccess = clientConnection.InjectToSend(rewrittenLoginResponsePacket);
          if (!injectSuccess) {
            throw std::runtime_error("Failed to inject rewritten login response packet to client connection");
          }
          packetLogger.logPacket(rewrittenLoginResponsePacket, /*blocked=*/false, PacketContainer::Direction::kBotToClient);
          while (clientConnection.security->HasPacketToSend()) {
            clientConnection.Send(clientConnection.security->GetPacketToSend());
          }
        }

        // Close active connections
        VLOG(2) << "Closing gateway connection, connecting to agent server";
        if (!clientless_) {
          clientConnection.Close();
        }
        serverConnection.Close();

        // Want to forward this to the bot so it can grab the token
        packetBroker_.packetReceived(packet, direction);

        if (clientless_) {
          ioService_.post(boost::bind(&Proxy::connectClientless, this));
        }

        // Security pointer is now invalid so skip to the end
        break;
      }
    } else if (clientless_ && static_cast<packet::Opcode>(packet.opcode) == packet::Opcode::kFrameworkStateRequest) {
      if (!connectToAgent_) {
        VLOG(2) << "Received framework state request, sending patch request";
        const PacketContainer patchRequestPacket = packet::building::ClientGatewayPatchRequest::packet(divisionInfo_.locale, "SR_Client", /*version=*/188);

        packetLogger.logPacket(patchRequestPacket, /*blocked=*/false, PacketContainer::Direction::kBotToServer);
        serverConnection.InjectToSend(patchRequestPacket);
        // Inject the packet immediately
        while (serverConnection.security->HasPacketToSend()) {
          serverConnection.Send(serverConnection.security->GetPacketToSend());
        }
        VLOG(2) << "Sent gateway patch request response";
      }
    }

    const auto opcodeAsEnum = static_cast<packet::Opcode>(packet.opcode);
    if (opcodeAsEnum == packet::Opcode::kServerAgentCharacterDataBegin) {
      // Initialize data/container
      if (characterInfoPacketContainer_) {
        // What? There's already one?
        LOG(INFO) << "[@@@] Wait, we got a char info begin packet, but we've already initialized the data";
      }
      // Initialize packet data with the "begin" data
      characterInfoPacketContainer_.emplace(packet);
      // Update opcode to reflect "data"
      characterInfoPacketContainer_->opcode = static_cast<uint16_t>(packet::Opcode::kServerAgentCharacterData);
    } else if (opcodeAsEnum == packet::Opcode::kServerAgentEntityGroupspawnBegin) {
      // Initialize data/container
      if (groupSpawnPacketContainer_) {
        // What? There's already one?
        LOG(INFO) << "[@@@] Wait, we got a char info begin packet, but we've already initialized the data";
      }
      // Initialize packet data with the "begin" data
      groupSpawnPacketContainer_.emplace(packet);
      // Update opcode to reflect "data"
      groupSpawnPacketContainer_->opcode = static_cast<uint16_t>(packet::Opcode::kServerAgentEntityGroupspawnData);
    } else if (opcodeAsEnum == packet::Opcode::kServerAgentInventoryStorageBegin) {
      // Initialize data/container
      if (storagePacketContainer_) {
        // What? There's already one?
        LOG(INFO) << "[@@@] Wait, we got a storage begin packet, but we've already initialized the data";
      }
      // Initialize packet data with the "begin" data
      storagePacketContainer_.emplace(packet);
      // Update opcode to reflect "data"
      storagePacketContainer_->opcode = static_cast<uint16_t>(packet::Opcode::kServerAgentInventoryStorageData);
    } else if (opcodeAsEnum == packet::Opcode::kServerAgentGuildStorageBegin) {
      // Initialize data/container
      if (guildStoragePacketContainer_) {
        // What? There's already one?
        LOG(INFO) << "[@@@] Wait, we got a guild storage begin packet, but we've already initialized the data";
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
    if (opcodeAsEnum == packet::Opcode::kServerAgentCharacterDataEnd) {
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
    } else if (opcodeAsEnum != packet::Opcode::kServerAgentCharacterDataBegin &&
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

    // Forward the packet to the Silkroad client, if there is one
    if (!clientless_) {
      if (forward) {
        if (!clientConnection.security) {
          throw std::runtime_error("Want to forward packet from server to client, but have no connection to the client (not in clientless mode)");
        }
        clientConnection.InjectToSend(packet);
      }
    }
  }
}

void Proxy::sendPacketsToServer() {
  if (!serverConnection.security) {
    // No server connection, nobody to send packets to
    return;
  }
  // Send packets that are currently in the security api
  while (serverConnection.security->HasPacketToSend()) {
    if (!serverConnection.Send(serverConnection.security->GetPacketToSend())) {
      LOG(ERROR) << "Server connection Send error.";
      break;
    }
    // If we ever need to run clientless, we need to know when the last packet was sent.
    lastPacketSentToServer_ = std::chrono::steady_clock::now();
  }
}