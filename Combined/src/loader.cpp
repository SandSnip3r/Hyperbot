#include "loader.hpp"
#include "../../common/Common.h"

#include <fstream>
#include <iostream>
#include <sstream>
// Silkroad path
// locale
// Gateway port

void createAppDataDirectoryIfNecessary(const std::experimental::filesystem::v1::path &appDataDirectoryPath) {
  const std::string kAppDataSubdirName = "Hyperbot"; // TODO: Move to a shared location since this is used in the DLL too
  if (!std::experimental::filesystem::v1::exists(appDataDirectoryPath / kAppDataSubdirName)) {
    std::experimental::filesystem::v1::create_directory(appDataDirectoryPath / kAppDataSubdirName);
  }
}

Loader::Loader(const std::experimental::filesystem::v1::path &kSilkroadDirectoryPath, const pk2::DivisionInfo &divisionInfo) : kSilkroadDirectoryPath_(kSilkroadDirectoryPath), kDivisionInfo_(divisionInfo) {
  // TODO: Ensure this dll path is updated for release builds
  dllPath_ = edxLabs::GetAbsoluteDirectoryPath() + "../Debug/loaderDll.dll";
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
  std::cout << "Client (PID:" << pi.dwProcessId << ") launched with arguments \"" << arguments_ << "\"\n";
  std::cout << "The client should connect to port " << proxyListeningPort << '\n';
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
    const std::experimental::filesystem::v1::path portInfoFilename = appDataDirectoryPath / kAppDataSubdirName / (std::to_string(pi.dwProcessId)+".txt");
    std::ofstream portInfoFile(portInfoFilename);
    if (portInfoFile) {
      portInfoFile << proxyListeningPort << '\n';
      std::cout << "Created file with port info\n";
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
  std::cout << "Successfully injected DLL\n";

  // Finally resume the client.
  ResumeThread(pi.hThread);
  ResumeThread(pi.hProcess);
  WSACleanup();
}

// void Loader::killClient() {}