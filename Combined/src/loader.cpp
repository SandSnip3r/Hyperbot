#include "loader.hpp"
#include "../../common/Common.h"

#include <iostream>
#include <sstream>

Loader::Loader(const std::string &silkroadPath) : kSilkroadPath(silkroadPath) {
  dllPath = edxLabs::GetAbsoluteDirectoryPath() + "../Debug/loaderDll.dll";
  sroData.path = kSilkroadPath;
  if (!LoadPath(sroData.path, sroData)) {
    throw std::runtime_error("Unable to load Silkroad data from \""+sroData.path+"\"");
  }
  std::stringstream args;
  args << "0 /" << (int)sroData.divInfo.locale << " " << 0 << " " << 0;
  arguments = args.str();
  std::string pathToClient = sroData.path;
  //TODO: Use a better filesystem utility
  clientPath = kSilkroadPath + "sro_client.exe";
  //TODO: Check this exists
  std::cout << "Loader constructed\n";
  std::cout << " Silkroad client path: \"" << clientPath << "\"\n";
  std::cout << " DLL path: \"" << dllPath << "\"\n";
  std::cout << " Gateway port " << sroData.gatePort << '\n';
}

void Loader::startClient() {
  WSADATA wsaData = { 0 };
  WSAStartup(MAKEWORD(2, 2), &wsaData);
  STARTUPINFOA si = { 0 };
  PROCESS_INFORMATION pi = { 0 };

  // Launch the client in a suspended state so we can patch it
  bool result = edxLabs::CreateSuspendedProcess(clientPath, arguments, si, pi);
  if (result == false) {
    throw std::runtime_error("Could not start \""+clientPath+"\"");
  }
  std::cout << "Client launched with arguments \"" << arguments << "\"\n";

  // Inject the DLL so we can have some fun
  result = (FALSE != edxLabs::InjectDLL(pi.hProcess, dllPath.c_str(), "OnInject", static_cast<DWORD>(edxLabs::GetEntryPoint(clientPath.c_str())), false));
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

void Loader::killClient() {

}