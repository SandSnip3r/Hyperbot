#include "clientManager.hpp"

#include <ui_proto/client_manager_request.pb.h>

#include <silkroad_lib/file_util.hpp>
#include <silkroad_lib/pk2/parsing/parsing.hpp>
#include <silkroad_lib/pk2/pk2.hpp>
#include <silkroad_lib/pk2/pk2ReaderModern.hpp>

#include <boost/dll/runtime_symbol_info.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <fstream>

ClientManager::ClientManager(std::string_view hyperbotIpAddress, int32_t hyperbotPort, std::string_view clientDirectoryPath) : hyperbotIpAddress_(hyperbotIpAddress), hyperbotPort_(hyperbotPort), clientDirectoryPath_(clientDirectoryPath) {
}

void ClientManager::run() {
  checkDllPath();
  checkClientPath();
  parseGameFiles();
  buildArguments();

  socket_.connect(absl::StrFormat("tcp://%s:%d", hyperbotIpAddress_, hyperbotPort_));
  VLOG(1) << absl::StreamFormat("ClientManager running. Connected to Hyperbot @ %s:%d", hyperbotIpAddress_, hyperbotPort_);

  while (1) {
    zmq::message_t request;
    VLOG(4) << "Waiting for request";
    std::vector<zmq::pollitem_t> pollItems = { { socket_, 0, ZMQ_POLLIN, 0 } };
    // Wait up to a certain amount of time for a message.
    int eventCount = zmq::poll(pollItems, kMissedHeartbeatTimeout_);
    if (eventCount == 0) {
      // No message was received within the specified timeout. It seems like the bot process is no longer running.
      if (!clientIdToProcessInformationMap_.empty()) {
        LOG(INFO) << "Bot seems to have disconnected. Killing " << clientIdToProcessInformationMap_.size() << " open clients.";
        killAllClients();
      }
      continue;
    }
    // Successfully received a message
    std::optional<zmq::recv_result_t> result = socket_.recv(request, zmq::recv_flags::none);
    if (!result) {
      throw std::runtime_error("Failed to receive from socket");
    }

    handleRequest(request);
  }
}

void ClientManager::handleRequest(const zmq::message_t &request) {
  proto::client_manager_request::Request requestProto;
  bool successfullyParsed = requestProto.ParseFromArray(request.data(), request.size());
  if (!successfullyParsed) {
    const std::string error = absl::StrFormat("Failed to parse request \"%s\"", request.to_string());
    LOG(WARNING) << error;
    replyWithError(error);
    return;
  }

  // Successfully parsed request.
  switch (requestProto.body_case()) {
    case proto::client_manager_request::Request::BodyCase::kStartClient: {
      VLOG(4) << "Received StartClient \"" << requestProto.DebugString() << '"';
      const proto::client_manager_request::StartClient &packet = requestProto.start_client();
      LOG(INFO) << "Port to connect to: " << packet.port_to_connect_to();
      const int32_t clientId = launchClient(packet.port_to_connect_to());
      proto::client_manager_request::ClientStarted clientStarted;
      clientStarted.set_client_id(clientId);
      proto::client_manager_request::Response response;
      response.mutable_client_started()->CopyFrom(clientStarted);
      std::optional<zmq::send_result_t> result = socket_.send(zmq::message_t(response.SerializeAsString()), zmq::send_flags::none);
      if (!result) {
        throw std::runtime_error("Failed to send response for StartClient");
      }
      break;
    }
    case proto::client_manager_request::Request::BodyCase::kHeartbeat: {
      VLOG(5) << "Received Heartbeat";
      proto::client_manager_request::Response response;
      const std::vector<ClientId> deadClientIds = reapDeadClients();
      if (deadClientIds.empty()) {
        // If no clients have died, we send a simple heartbeat_ack.
        response.mutable_heartbeat_ack();
      } else {
        // If clients have died, we'll respond to the heartbeat with a list of dead clients.
        proto::client_manager_request::ClientsDied *clientsDied = response.mutable_clients_died();
        for (ClientId deadClientId : deadClientIds) {
          clientsDied->add_client_ids(deadClientId);
        }
      }
      std::optional<zmq::send_result_t> result = socket_.send(zmq::message_t(response.SerializeAsString()), zmq::send_flags::none);
      if (!result) {
        throw std::runtime_error("Failed to send response for Heartbeat");
      }
      break;
    }
    default: {
      const std::string error = absl::StrFormat("Unknown message body case %d", static_cast<int>(requestProto.body_case()));
      LOG(WARNING) << error;
      replyWithError(error);
      break;
    }
  }
}

void ClientManager::replyWithError(const std::string &errorMessage) {
  proto::client_manager_request::Error error;
  error.set_message(errorMessage);
  proto::client_manager_request::Response response;
  response.mutable_error()->CopyFrom(error);
  std::optional<zmq::send_result_t> result = socket_.send(zmq::message_t(response.SerializeAsString()), zmq::send_flags::none);
  if (!result) {
    throw std::runtime_error("Failed to send error message");
  }
}

void ClientManager::checkDllPath() {
  boost::dll::fs::path exePath = boost::dll::program_location();
  boost::dll::fs::path exeDir = exePath.parent_path();
  dllPath_ = std::filesystem::path(exeDir.string()) / "loader_dll.dll";
  if (!std::filesystem::exists(dllPath_)) {
    throw std::runtime_error(absl::StrFormat("loader_dll.dll does not exist at \"%s\"", dllPath_.string()));
  }
  VLOG(1) << "Found loader_dll.dll at " << dllPath_;
}
void ClientManager::checkClientPath() {
  clientPath_ = std::filesystem::path(clientDirectoryPath_) / "sro_client.exe";
  if (!std::filesystem::exists(clientPath_)) {
    throw std::runtime_error(absl::StrFormat("sro_client.exe does not exist at \"%s\"", clientPath_.string()));
  }
  VLOG(1) << "Found sro_client.exe at " << clientPath_;
}

void ClientManager::parseGameFiles() {
  LOG(INFO) << "Parsing game files at " << clientDirectoryPath_;
  sro::pk2::Pk2ReaderModern pk2Reader{clientDirectoryPath_ / "Media.pk2"};
  const std::string kDivisionInfoEntryName = "DIVISIONINFO.TXT";
  sro::pk2::PK2Entry divisionInfoEntry = pk2Reader.getEntry(kDivisionInfoEntryName);
  const std::vector<uint8_t> divisionInfoData = pk2Reader.getEntryData(divisionInfoEntry);
  sro::pk2::DivisionInfo divisionInfo = sro::pk2::parsing::parseDivisionInfo(divisionInfoData);
  locale_ = divisionInfo.locale;
  LOG(INFO) << "Parsed locale as " << locale_;
}

void ClientManager::buildArguments() {
  arguments_ = absl::StrFormat("0 /%d 0 0", locale_);
  VLOG(3) << absl::StreamFormat("Client arguments are \"%s\"", arguments_);
}

int32_t ClientManager::launchClient(int32_t portToConnectTo) {
  WSADATA wsaData = { 0 };
  WSAStartup(MAKEWORD(2, 2), &wsaData);
  STARTUPINFOA si = { 0 };
  PROCESS_INFORMATION processInformation = { 0 };

  // Launch the client in a suspended state so we can patch it
  bool result = sro::edx_labs::CreateSuspendedProcess(clientPath_.string(), arguments_, si, processInformation);
  if (result == false) {
    throw std::runtime_error("Could not start \""+clientPath_.string()+"\"");
  }
  VLOG(1) << "Client " << clientPath_ << " (PID:" << processInformation.dwProcessId << ") launched with arguments \"" << arguments_ << '"';
  VLOG(1) << "The client should connect to port " << portToConnectTo;
  {
    // Write to a file (<Client PID>.txt) the port that the client should connect to
    // TODO: Replace %APPDATA% with %TEMP% to prevent stray file buildup
    const auto appDataDirectoryPath = sro::file_util::getAppDataPath();
    if (appDataDirectoryPath.empty()) {
      throw std::runtime_error("Unable to find %APPDATA%\n");
    }
    const std::filesystem::path portInfoFilename = appDataDirectoryPath / (std::to_string(processInformation.dwProcessId)+".txt");
    std::ofstream portInfoFile(portInfoFilename);
    if (portInfoFile) {
      portInfoFile << portToConnectTo << '\n';
    } else {
      throw std::runtime_error("Unable to open file \"" + portInfoFilename.string() + "\" to communicate port to DLL\n");
    }
  }

  // Inject the DLL so we can have some fun
  result = (FALSE != sro::edx_labs::InjectDLL(processInformation.hProcess, dllPath_.string().c_str(), "OnInject", static_cast<DWORD>(sro::edx_labs::GetEntryPoint(clientPath_.string().c_str())), false));
  if (result == false) {
    TerminateThread(processInformation.hThread, 0);
    throw std::runtime_error("Could not inject into the Silkroad client process");
  }

  // Finally resume the client.
  ResumeThread(processInformation.hThread);
  ResumeThread(processInformation.hProcess);
  WSACleanup();

  ClientId clientId = saveProcessInformation(processInformation);
  return clientId;
}

int32_t ClientManager::saveProcessInformation(PROCESS_INFORMATION processInformation) {
  ClientId clientId = nextClientId_;
  clientIdToProcessInformationMap_[clientId] = processInformation;
  VLOG(3) << "Saving process ID " << processInformation.hProcess << " as client ID " << clientId;
  ++nextClientId_;
  return clientId;
}

std::vector<ClientManager::ClientId> ClientManager::reapDeadClients() {
  std::vector<ClientId> deadClientIds;
  for (const auto [id, processInformation] : clientIdToProcessInformationMap_) {
    // Poll the process with no wait to quickly see if it has exited.
    if (WaitForSingleObject(processInformation.hProcess, /*dwMilliseconds=*/0) == WAIT_OBJECT_0) {
      LOG(INFO) << "Client " << id << " (pid:" << processInformation.dwProcessId << ") has exited";
      deadClientIds.emplace_back(id);
    }
  }
  for (ClientId deadClientId : deadClientIds) {
    clientIdToProcessInformationMap_.erase(deadClientId);
  }
  return deadClientIds;
}

void ClientManager::killAllClients() {
  for (const auto [id, processInformation] : clientIdToProcessInformationMap_) {
    HANDLE handle = OpenProcess(PROCESS_TERMINATE, FALSE, processInformation.dwProcessId);
    if (handle == NULL) {
      // Unable to "open" process?
    } else {
      if (TerminateProcess(handle, -1)) {
        // Success!
        VLOG(1) << "Successfully killed client " << id;
      } else {
        // Failed...?
        VLOG(1) << "Failed to kill client " << id;
      }
      CloseHandle(handle);
    }
  }
  clientIdToProcessInformationMap_.clear();
}