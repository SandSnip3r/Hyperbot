#include "loader.hpp"
#include "../../common/Common.h"

#include <iostream>
#include <sstream>

// Silkroad path
// locale
// Gateway port

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

void Loader::startClient() {
  WSADATA wsaData = { 0 };
  WSAStartup(MAKEWORD(2, 2), &wsaData);
  STARTUPINFOA si = { 0 };
  PROCESS_INFORMATION pi = { 0 };

  // Launch the client in a suspended state so we can patch it
  bool result = edxLabs::CreateSuspendedProcess(clientPath_, arguments_, si, pi);
  if (result == false) {
    throw std::runtime_error("Could not start \""+clientPath_+"\"");
  }
  std::cout << "Client launched with arguments \"" << arguments_ << "\"\n";

  // Inject the DLL so we can have some fun
  result = (FALSE != edxLabs::InjectDLL(pi.hProcess, dllPath_.c_str(), "OnInject", static_cast<DWORD>(edxLabs::GetEntryPoint(clientPath_.c_str())), false));
  if (result == false) {
    TerminateThread(pi.hThread, 0);
    throw std::runtime_error("Could not inject into the Silkroad client process");
  }
  std::cout << "Successfully injected DLL into process with id: " << pi.dwProcessId << "\n";

  // Finally resume the client.
  ResumeThread(pi.hThread);
  ResumeThread(pi.hProcess);
  WSACleanup();
}

// void Loader::killClient() {}