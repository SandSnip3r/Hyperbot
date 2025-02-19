#ifndef CLIENTMANAGER_HPP_
#define CLIENTMANAGER_HPP_

#include <silkroad_lib/edx_labs.hpp>
#include <silkroad_lib/pk2/divisionInfo.hpp>

#include <zmq.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>

class ClientManager {
public:
  ClientManager(std::string_view hyperbotIpAddress, int32_t hyperbotPort, std::string_view clientDirectoryPath);
  void run();
private:
  using ClientId = int32_t;

  // In the bot, the heartbeat should be sent every 100ms. This timeout allows for a 3x margin of error.
  static constexpr std::chrono::milliseconds kMissedHeartbeatTimeout_{300};

  zmq::context_t context_;
  zmq::socket_t socket_{context_, zmq::socket_type::rep};
  const std::string hyperbotIpAddress_;
  const int32_t hyperbotPort_;
  const std::filesystem::path clientDirectoryPath_;

  std::filesystem::path dllPath_;
  std::filesystem::path clientPath_;
  // From Media.pk2's divisioninfo.txt
  int locale_;
	std::string arguments_;

  std::map<ClientId, DWORD> clientIdToProcessIdMap_;
  ClientId nextClientId_{0};

  void checkDllPath();
  void checkClientPath();
  void parseGameFiles();
  void buildArguments();
  int32_t launchClient(int32_t portToConnectTo);
  void replyWithError(const std::string &errorMessage);

  // Takes the actual process ID and returns a unique "ClientId" which can be shared.
  int32_t saveProcessId(DWORD processId);

  void killAllClients();
};

#endif // CLIENTMANAGER_HPP_