#ifndef PROXY_HPP_
#define PROXY_HPP_

#include "packetLogger.hpp"
#include "silkroadConnection.hpp"
#include "pk2/gameData.hpp"
#include "shared/silkroad_security.h"

#include <silkroad_lib/pk2/divisionInfo.hpp>

#include <boost/make_shared.hpp>
#include <boost/asio.hpp>

#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

class PacketProcessor;

// Networking class (handles connections)
class Proxy {
public:
  Proxy(const pk2::GameData &gameData, PacketProcessor &processor, uint16_t port = 0);
  ~Proxy();
  void inject(const PacketContainer &packet, const PacketContainer::Direction direction);
  void runAsync();
  uint16_t getOurListeningPort() const;
  void blockOpcode(packet::Opcode opcode);
  void unblockOpcode(packet::Opcode opcode);
  [[nodiscard]] bool blockingOpcode(packet::Opcode opcode) const;
  bool isClientless() const { return clientless_; }
  void setClientless(bool clientless);

  void connectClientlessAsync();
  void connectClientless();

  // Stops all networking objects
  void stop();
private:
  static constexpr int kMillisecondsBetweenKeepalives{5000};
  uint16_t ourListeningPort_;
  uint16_t gatewayPort_;
  const sro::pk2::DivisionInfo divisionInfo_;
  PacketProcessor &packetProcessor_;
  std::string gatewayAddress_;
  std::string agentIP_;
  uint16_t agentPort_{0};
  std::set<uint16_t> blockedOpcodes_;
  bool connectToAgent_{false};
  boost::asio::io_service ioService_;
  const int kPacketProcessDelayMs{10};
  PacketLogger packetLogger{"C:\\Users\\Victor\\Documents\\Development\\packet-logs\\"};
  std::optional<PacketContainer> characterInfoPacketContainer_, groupSpawnPacketContainer_, storagePacketContainer_, guildStoragePacketContainer_;
  std::thread thr_;
  std::atomic<bool> clientless_{false};
  std::chrono::steady_clock::time_point lastPacketSentToServer_{std::chrono::steady_clock::now()};
  boost::shared_ptr<boost::asio::steady_timer> keepAlivePacketTimer_;
  std::deque<PacketContainer> injectedClientPacketsForClientless_;

  //Accepts TCP connections
  boost::asio::ip::tcp::acceptor acceptor;

  // Packet processing timer
  boost::shared_ptr<boost::asio::steady_timer> packetProcessingTimer_;

  //Silkroad connections
  SilkroadConnection clientConnection{ioService_, "Client"};
  SilkroadConnection serverConnection{ioService_, "Server"};

  //Starts accepting new connections
  void PostAccept();

  //Handles new connections
  void HandleAccept(boost::shared_ptr<boost::asio::ip::tcp::socket> s, const boost::system::error_code & error);

  void ProcessPackets(const boost::system::error_code &error);
  void setKeepaliveTimer();
  void checkClientlessKeepalive(const boost::system::error_code &error);

  void receivePacketsFromClient();
  void sendPacketsToClient();
  void receivePacketsFromServer();
  void sendPacketsToServer();

  void run();
};

#endif
