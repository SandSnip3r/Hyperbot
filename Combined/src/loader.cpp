#include "loader.hpp"
#include "logging.hpp"

#include "../../common/Common.h"

#include <csignal>
#include <functional>
#include <fstream>
#include <iostream>
#include <sstream>

void createAppDataDirectoryIfNecessary(const std::filesystem::path &appDataDirectoryPath) {
  const std::string kAppDataSubdirName = "Hyperbot"; // TODO: Move to a shared location since this is used in the DLL too
  if (!std::filesystem::exists(appDataDirectoryPath / kAppDataSubdirName)) {
    std::filesystem::create_directory(appDataDirectoryPath / kAppDataSubdirName);
  }
}

Loader::Loader(const std::filesystem::path &kSilkroadDirectoryPath, const pk2::DivisionInfo &divisionInfo) : kSilkroadDirectoryPath_(kSilkroadDirectoryPath), kDivisionInfo_(divisionInfo) {
  // TODO: Ensure this dll path is updated for release builds
  // Note: Assuming that the DLL is in our current directory
  // TODO: Replace edx directory helper with std::fs
  dllPath_ = edxLabs::GetAbsoluteDirectoryPath() + "/loaderDll.dll";
  std::stringstream args;
  args << "0 /" << (int)kDivisionInfo_.locale << " " << 0 << " " << 0;
  arguments_ = args.str();
  //TODO: Use a better filesystem utility
  clientPath_ = (kSilkroadDirectoryPath_ / "sro_client.exe").string();
  //TODO: Check this exists
  std::cout << "Loader constructed\n";
  std::cout << " Silkroad client path: \"" << clientPath_ << "\"\n";
  std::cout << " DLL path: \"" << dllPath_ << "\"\n";
}

namespace {

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

} // anonymous namespace

void Loader::startClient(uint16_t proxyListeningPort) {
  WSADATA wsaData = { 0 };
  WSAStartup(MAKEWORD(2, 2), &wsaData);
  STARTUPINFOA si = { 0 };
  PROCESS_INFORMATION pi = { 0 };

  // Launch the client in a suspended state so we can patch it
  bool result = edxLabs::CreateSuspendedProcess(clientPath_, arguments_, si, pi);
  if (result == false) {
    throw std::runtime_error("Could not start \""+clientPath_+"\"");
  }
  LOG() << "Client (PID:" << pi.dwProcessId << ") launched with arguments \"" << arguments_ << '"' << std::endl;
  LOG() << "The client should connect to port " << proxyListeningPort << std::endl;
  {
    // Write to a file (<Client PID>.txt) the port that the client should connect to
    // TODO: Replace %APPDATA% with %TEMP% to prevent stray file buildup
    const auto appDataDirectoryPath = getAppDataPath();
    if (appDataDirectoryPath.empty()) {
      throw std::runtime_error("Unable to find %APPDATA%\n");
    }
    // Make sure the output directory exists
    createAppDataDirectoryIfNecessary(appDataDirectoryPath);
    const std::string kAppDataSubdirName = "Hyperbot"; // TODO: Move to a shared location since this is used in the DLL too
    const std::filesystem::path portInfoFilename = appDataDirectoryPath / kAppDataSubdirName / (std::to_string(pi.dwProcessId)+".txt");
    std::ofstream portInfoFile(portInfoFilename);
    if (portInfoFile) {
      portInfoFile << proxyListeningPort << '\n';
    } else {
      throw std::runtime_error("Unable to open file \"" + portInfoFilename.string() + "\" to communicate port to DLL\n");
    }
  }

  // Inject the DLL so we can have some fun
  result = (FALSE != edxLabs::InjectDLL(pi.hProcess, dllPath_.c_str(), "OnInject", static_cast<DWORD>(edxLabs::GetEntryPoint(clientPath_.c_str())), false));
  if (result == false) {
    TerminateThread(pi.hThread, 0);
    throw std::runtime_error("Could not inject into the Silkroad client process");
  }
  LOG() << "Successfully injected DLL" << std::endl;

  // Finally resume the client.
  ResumeThread(pi.hThread);
  ResumeThread(pi.hProcess);
  WSACleanup();

  // Kill the process when we exit
  Murderer::addProcessToKillOnExit(pi.dwProcessId);
  signal(SIGINT, &Murderer::signalHandler);
}

// void Loader::killClient() {}