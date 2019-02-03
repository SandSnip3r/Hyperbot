#include "loader.hpp"
#include "../../common/Common.h"

#include <iostream>
#include <sstream>

Loader::Loader(const std::string &silkroadPath) : kSilkroadPath(silkroadPath) {
  dllPath = edxLabs::GetAbsoluteDirectoryPath() + "../Debug/loaderDll.dll";
  sroData.path = kSilkroadPath;
  LoadPath(sroData.path, sroData); //TODO: Check result
  std::stringstream args;
  args << "0 /" << (int)sroData.divInfo.locale << " " << 0 << " " << 0;
  arguments = args.str();
  std::string pathToClient = sroData.path;
  //TODO: Use a better filesystem utility
  clientPath = kSilkroadPath + "sro_client.exe";
  //TODO: Check this exists
}

void Loader::startClient() {
  WSADATA wsaData = { 0 };
  WSAStartup(MAKEWORD(2, 2), &wsaData);
  STARTUPINFOA si = { 0 };
  PROCESS_INFORMATION pi = { 0 };

  // Launch the client in a suspended state so we can patch it
  bool result = edxLabs::CreateSuspendedProcess(clientPath, arguments, si, pi);
  if (result == false) {
    throw std::runtime_error("Could not start \"sro_client.exe\"");
  }
  std::cout << "Launched client\n";

  // Inject the DLL so we can have some fun
  result = (FALSE != edxLabs::InjectDLL(pi.hProcess, dllPath.c_str(), "OnInject", static_cast<DWORD>(edxLabs::GetEntryPoint(clientPath.c_str())), false));
  if (result == false) {
    TerminateThread(pi.hThread, 0);
    throw std::runtime_error("Could not inject into the Silkroad client process");
  }
  std::cout << "Injected DLL\n";

  // Finally resume the client.
  ResumeThread(pi.hThread);
  ResumeThread(pi.hProcess);
  WSACleanup();
}

void Loader::killClient() {
  //TODO
}