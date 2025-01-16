#include "loader.hpp"

#include <silkroad_lib/edx_labs.h>
#include <silkroad_lib/file_util.h>

#include <absl/log/log.h>

#include <csignal>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

Loader::Loader(std::string_view clientPath, const pk2::DivisionInfo &divisionInfo) : kDivisionInfo_(divisionInfo) {
#if defined(_WIN32)
  // TODO: Ensure this dll path is updated for release builds
  // Note: We assume that the DLL is in our current directory
  dllPath_ = std::filesystem::current_path() / "loader_dll.dll";
  if (!std::filesystem::exists(dllPath_)) {
    throw std::runtime_error("loader_dll.dll does not exist next to executable");
  }
  clientPath_ = std::filesystem::path(clientPath) / "sro_client.exe";
  std::stringstream args;
  args << "0 /" << (int)kDivisionInfo_.locale << " " << 0 << " " << 0;
  arguments_ = args.str();
  if (!std::filesystem::exists(clientPath_)) {
    throw std::runtime_error("sro_client.exe does not exist");
  }
#endif
}

namespace {

#if defined(_WIN32)

class Murderer {
public:
  static std::vector<int> processIds_;
  static void addProcessToKillOnExit(int processId) {
    processIds_.push_back(processId);
  }
  static void signalHandler(int signal) {
    // https://ladydebug.com/blog/2019/08/29/kill-process-programmatically-by-pid/
    for (auto pid : processIds_) {
      HANDLE handle = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
      if (handle == NULL) {
        // Unable to "open" process?
      } else {
        if (TerminateProcess(handle, -1)) {
          // Success!
        } else {
          // Failed...?
        }
        CloseHandle(handle);
      }
    }
    exit(0);
  }
};
std::vector<int> Murderer::processIds_;

#endif

} // anonymous namespace

void Loader::startClient(uint16_t proxyListeningPort) {
#if defined(_WIN32)
  WSADATA wsaData = { 0 };
  WSAStartup(MAKEWORD(2, 2), &wsaData);
  STARTUPINFOA si = { 0 };
  PROCESS_INFORMATION pi = { 0 };

  // Launch the client in a suspended state so we can patch it
  bool result = sro::edx_labs::CreateSuspendedProcess(clientPath_.string(), arguments_, si, pi);
  if (result == false) {
    throw std::runtime_error("Could not start \""+clientPath_.string()+"\"");
  }
  VLOG(1) << "Client " << clientPath_ << " (PID:" << pi.dwProcessId << ") launched with arguments \"" << arguments_ << '"';
  VLOG(1) << "The client should connect to port " << proxyListeningPort;
  {
    // Write to a file (<Client PID>.txt) the port that the client should connect to
    // TODO: Replace %APPDATA% with %TEMP% to prevent stray file buildup
    const auto appDataDirectoryPath = sro::file_util::getAppDataPath();
    if (appDataDirectoryPath.empty()) {
      throw std::runtime_error("Unable to find %APPDATA%\n");
    }
    const std::filesystem::path portInfoFilename = appDataDirectoryPath / (std::to_string(pi.dwProcessId)+".txt");
    std::ofstream portInfoFile(portInfoFilename);
    if (portInfoFile) {
      portInfoFile << proxyListeningPort << '\n';
    } else {
      throw std::runtime_error("Unable to open file \"" + portInfoFilename.string() + "\" to communicate port to DLL\n");
    }
  }

  // Inject the DLL so we can have some fun
  result = (FALSE != sro::edx_labs::InjectDLL(pi.hProcess, dllPath_.string().c_str(), "OnInject", static_cast<DWORD>(sro::edx_labs::GetEntryPoint(clientPath_.string().c_str())), false));
  if (result == false) {
    TerminateThread(pi.hThread, 0);
    throw std::runtime_error("Could not inject into the Silkroad client process");
  }

  // Finally resume the client.
  ResumeThread(pi.hThread);
  ResumeThread(pi.hProcess);
  WSACleanup();

  // Kill the process when we exit
  Murderer::addProcessToKillOnExit(pi.dwProcessId);
  signal(SIGINT, &Murderer::signalHandler);
#else
  throw std::runtime_error("Cannot start client on non-Windows systems");
#endif
}

// void Loader::killClient() {}