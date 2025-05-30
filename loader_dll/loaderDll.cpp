// edxSilkroadDll_Lite

// This project based on my Silkroad development series guides:
// http://www.elitepvpers.de/forum/sro-guides-templates/271405-guide-silkroad-development-series.html

//-------------------------------------------------------------------------

#include "detours/detours.h"

#include <silkroad_lib/dll_config.hpp>
#include <silkroad_lib/edx_labs.hpp>
#include <silkroad_lib/file_util.hpp>
#include <silkroad_lib/pk2/divisionInfo.hpp>
#include <silkroad_lib/pk2/pk2ReaderModern.hpp>
#include <silkroad_lib/pk2/parsing/parsing.hpp>

#include <windows.h>
#include <windowsx.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

//-------------------------------------------------------------------------

HINSTANCE gInstance = 0;
DWORD languageFlag = 0;
sro::pk2::DivisionInfo gDivisionInfo;
bool bDoLanguagePatch = true;
bool bDoMulticlient = true;
bool bDoZoomhack = true;
bool bDoNudePatch = true;
bool bDoSwearFilter = true;
bool bDoSecuritySeed = false;
bool bDoRedirectGateway = true;
bool bDoRedirectAgent = true;
bool bDebugConsole = true;
bool bHookInput = true;
bool bPatchKsroImgCode = false;
std::string defaultGatewayIP = "127.0.0.1";
std::string defaultAgentIP = "127.0.0.1";

//-------------------------------------------------------------------------

void UserOnInject(sro::edx_labs::DllConfig *config);
void UserOnUninject();
void OnConsoleInput(std::string input);

//-------------------------------------------------------------------------

template <typename type>
std::string ToString(const type & input)
{
  std::stringstream ss;
  ss << input;
  return ss.str();
}

// Main DLL entry point
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ulReason, LPVOID lpReserved)
{
  UNREFERENCED_PARAMETER(lpReserved);
  if (ulReason == DLL_PROCESS_ATTACH)
  {
    gInstance = hModule;
    // Do not notify this DLL of thread based events
    DisableThreadLibraryCalls(hModule);
  }
  else if (ulReason == DLL_PROCESS_DETACH)
  {
    UserOnUninject();
  }
  return TRUE;
}

//-------------------------------------------------------------------------

// This is the main function that is called when the DLL is injected into the process
extern "C" __declspec(dllexport) void OnInject(DWORD address, LPDWORD bytes) {
  // Restore the original bytes at the OEP
  DWORD wrote = 0;
  WriteProcessMemory(GetCurrentProcess(), UlongToPtr(address), bytes, 6, &wrote);

  sro::edx_labs::DllConfig *config = reinterpret_cast<sro::edx_labs::DllConfig*>(reinterpret_cast<uint8_t*>(bytes) + 6);
  if (sizeof(sro::edx_labs::DllConfig) != config->size) {
    // If the size is not correct, we cannot use the config
    const std::string errorMsg = "Config size mismatch: expected " + std::to_string(sizeof(sro::edx_labs::DllConfig)) + ", got " + std::to_string(config->size);
    MessageBox(0, errorMsg.c_str(), "Fatal Error", MB_ICONERROR);
    ExitProcess(1);
  }

  // Call our user function to keep this function clean
  UserOnInject(config);
}

//-------------------------------------------------------------------------

// Required detours function
HMODULE WINAPI Detoured()
{
  return gInstance;
}

//-------------------------------------------------------------------------

DWORD WINAPI InjectionThread(LPVOID lpParam);
namespace nsDetourLoadStringA
{
  extern "C" int (WINAPI * Real_LoadString)(HINSTANCE hInstance, UINT uID, LPSTR lpBuffer, int nBufferMax) = LoadStringA;

  int WINAPI Detour_LoadString(HINSTANCE hInstance, UINT uID, LPSTR lpBuffer, int nBufferMax)
  {
    //if(hInstance == (HINSTANCE)0x400000)
    //{
    if (uID == 0x67)
    {
      HANDLE hThread = CreateThread(0, 0, InjectionThread, hInstance, 0, 0);
      WaitForSingleObject(hThread, INFINITE);
    }
    //}
    return Real_LoadString(hInstance, uID, lpBuffer, nBufferMax);
  }
}

//-------------------------------------------------------------------------

namespace nsDetourConnect
{
  // How many routes we have to work with
  size_t routeListCount = 16;

  // Custom detour routing structure
#pragma pack(push, 1)
  struct TDetourRoute
  {
    BYTE srcA;
    BYTE srcB;
    BYTE srcC;
    BYTE srcD;
    WORD srcPort;

    BYTE dstA;
    BYTE dstB;
    BYTE dstC;
    BYTE dstD;
    WORD dstPort;
  };
#pragma pack(pop)

  // Pointer to the route list
  TDetourRoute routeArray[16] = { 0 };
  TDetourRoute * routeList = routeArray;

  extern "C" int (WINAPI * Real_connect)(SOCKET, const struct sockaddr*, int) = connect;

  int WINAPI Detour_connect(SOCKET s, const struct sockaddr* name, int namelen)
  {
    // Store the real port
    WORD port = ntohs((*(WORD*)name->sa_data));

    // Breakup the IP into the parts
    BYTE a = name->sa_data[2];
    BYTE b = name->sa_data[3];
    BYTE c = name->sa_data[4];
    BYTE d = name->sa_data[5];

    struct sockaddr myname = { 0 };
    memcpy(&myname, name, sizeof(sockaddr));
    // Loop through the vector of routed addresses
    for (size_t x = 0; x < routeListCount; ++x)
    {
      const TDetourRoute & route = routeList[x];
      // If the port matches or the port doesn't matter
      if (route.srcPort == port || route.srcPort == -1)
      {
        // If the addresses match
        if (route.srcA != 255 && route.srcA != a)
          continue;
        if (route.srcB != 255 && route.srcB != b)
          continue;
        if (route.srcC != 255 && route.srcC != c)
          continue;
        if (route.srcD != 255 && route.srcD != d)
          continue;

        // Use the new address instead!
        myname.sa_data[2] = (char)route.dstA;
        myname.sa_data[3] = (char)route.dstB;
        myname.sa_data[4] = (char)route.dstC;
        myname.sa_data[5] = (char)route.dstD;

        // If the dst port is -1, use the original port
        (*(WORD*)myname.sa_data) = htons(route.dstPort == -1 ? port : route.dstPort);

        // Detoured connect in effect
        return Real_connect(s, &myname, namelen);
      }
    }
    // Regular connect
    return Real_connect(s, name, namelen);
  }
}

//-------------------------------------------------------------------------

namespace nsEnglishPatch
{
  // Function we use to obtain the global settings object
  FARPROC GetGlobalSettingsFunc = 0;
  DWORD customLangAddr;
  DWORD codecave_EnglishPatch_ReturnAddress = 0;
  __declspec(naked) void codecave_EnglishPatch()
  {
    __asm pop codecave_EnglishPatch_ReturnAddress
    __asm pushad
    __asm pushfd
    __asm mov ecx, languageFlag
    __asm call GetGlobalSettingsFunc
    __asm MOV DWORD PTR DS : [EAX + 0x160], ecx
    __asm popfd
    __asm popad
    __asm mov eax, customLangAddr
    __asm CMP DWORD PTR DS : [eax], 0 // original code, needs updating each client patch
    __asm push codecave_EnglishPatch_ReturnAddress
    __asm ret
  }
}

//-------------------------------------------------------------------------

namespace nsMulticlient
{
  // Function we use to obtain the global settings object (needs updating each patch)
  FARPROC AppendBytesFunc = 0;

  LPBYTE pMac;
  void Multiclient()
  {
    *((LPDWORD)(pMac + 2)) = GetTickCount();
  }

  DWORD codecave_Multiclient_ReturnAddress = 0;
  __declspec(naked) void codecave_Multiclient()
  {
    __asm pop codecave_Multiclient_ReturnAddress
    __asm mov pMac, eax
    __asm pushad
    __asm pushfd
    Multiclient();
    __asm popfd
    __asm popad
    __asm call AppendBytesFunc // Original code
    __asm push codecave_Multiclient_ReturnAddress
    __asm ret
  }
}

//-------------------------------------------------------------------------

namespace nsHookInput
{
  wchar_t * message;
  DWORD hookInputCustomAddr = 0;

  void HookInput()
  {
    char mbMessage[4096] = { 0 };
    wcstombs(mbMessage, message, 4095);
    OnConsoleInput(mbMessage);
  }

  DWORD codecave_HookInput_ReturnAddress = 0;
  __declspec(naked) void codecave_HookInput()
  {
    __asm pop codecave_HookInput_ReturnAddress
    __asm mov message, EDI
    __asm pushad
    __asm pushfd
    __asm call HookInput
    __asm popfd
    __asm popad
    __asm push edi
    __asm mov edi, hookInputCustomAddr
    __asm cmp[edi], ebp
    __asm pop edi
    __asm push codecave_HookInput_ReturnAddress
    __asm ret
  }
}

//-------------------------------------------------------------------------

namespace nsEnglishCaptcha
{
  FARPROC AppendStringFunc = 0;

  char * pImageCode;
  char newImageCode[7];
  char * pNewImageCode;

  char b1[3] = { 0 };
  char b2[3] = { 0 };
  char b3[3] = { 0 };
  char b4[3] = { 0 };
  char b5[3] = { 0 };
  char b6[3] = { 0 };

  void EnglishCaptcha()
  {
    b1[0] = pImageCode[0];
    b1[1] = pImageCode[1];
    b1[2] = 0;
    newImageCode[0] = sro::edx_labs::HexStringToInteger(b1);

    b2[0] = pImageCode[2];
    b2[1] = pImageCode[3];
    b2[2] = 0;
    newImageCode[1] = sro::edx_labs::HexStringToInteger(b2);

    b3[0] = pImageCode[4];
    b3[1] = pImageCode[5];
    b3[2] = 0;
    newImageCode[2] = sro::edx_labs::HexStringToInteger(b3);

    b4[0] = pImageCode[6];
    b4[1] = pImageCode[7];
    b4[2] = 0;
    newImageCode[3] = sro::edx_labs::HexStringToInteger(b4);

    b5[0] = pImageCode[8];
    b5[1] = pImageCode[9];
    b5[2] = 0;
    newImageCode[4] = sro::edx_labs::HexStringToInteger(b5);

    b6[0] = pImageCode[10];
    b6[1] = pImageCode[11];
    b6[2] = 0;
    newImageCode[5] = sro::edx_labs::HexStringToInteger(b6);

    newImageCode[6] = 0;

    printf("Image Code: %.2X%.2X%.2X%.2X%.2X%.2X\n", (BYTE)newImageCode[0], (BYTE)newImageCode[1], (BYTE)newImageCode[2], (BYTE)newImageCode[3], (BYTE)newImageCode[4], (BYTE)newImageCode[5]);

    pNewImageCode = newImageCode;
  }

  DWORD codecave_Captcha_ReturnAddress = 0;
  __declspec(naked) void codecave_EnglishCaptcha()
  {
    __asm pop codecave_Captcha_ReturnAddress
    __asm mov pImageCode, eax
    __asm pushad
    __asm pushfd
    EnglishCaptcha();
    __asm popfd
    __asm popad
    __asm pop eax
    __asm mov eax, pNewImageCode
    __asm push eax
    __asm call AppendStringFunc // Original code
    __asm push codecave_Captcha_ReturnAddress
    __asm ret
  }
}

//-------------------------------------------------------------------------

void modifyRoutelist(sro::edx_labs::DllConfig *config) {
  if (bDoRedirectGateway) {
    // Redirecting gateway
    using namespace nsDetourConnect;

    // MessageBox(0, ("Redirecting gateway to " + defaultGatewayIP + " " + std::to_string(config->hyperbotPort)).c_str(), "Redirect", MB_OK);

    std::vector<std::string> tokens = sro::edx_labs::TokenizeString(defaultGatewayIP, " .");
    if (tokens.size() != 4) {
      MessageBox(0, "Please enter an IP in the format 0.0.0.0", "Fatal Error", MB_ICONERROR);
      return;
    }

    for (size_t x = 0; x < routeListCount; ++x) {
      routeArray[x].dstA = atoi(tokens[0].c_str());
      routeArray[x].dstB = atoi(tokens[1].c_str());
      routeArray[x].dstC = atoi(tokens[2].c_str());
      routeArray[x].dstD = atoi(tokens[3].c_str());
      routeArray[x].dstPort = config->hyperbotPort;
    }

    WSADATA wsaData = { 0 };
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    routeListCount = 0;

    for (size_t x = 0; x < gDivisionInfo.divisions.size(); ++x) {
      for (size_t y = 0; y < gDivisionInfo.divisions[x].gatewayIpAddresses.size(); ++y) {
        std::string nme = gDivisionInfo.divisions[x].gatewayIpAddresses[y];
        struct hostent * remoteHost = gethostbyname(nme.c_str());
        if (remoteHost) {
          struct in_addr addr;
          addr.s_addr = *(u_long *)remoteHost->h_addr_list[0];
          std::string hostip = inet_ntoa(addr);
          tokens = sro::edx_labs::TokenizeString(hostip, ".");
          routeArray[routeListCount].srcA = atoi(tokens[0].c_str());
          routeArray[routeListCount].srcB = atoi(tokens[1].c_str());
          routeArray[routeListCount].srcC = atoi(tokens[2].c_str());
          routeArray[routeListCount].srcD = atoi(tokens[3].c_str());

          routeArray[routeListCount].srcPort = config->gatewayPort; // rather than doing another GATEPORT access

          routeListCount++;
        } else {
          MessageBox(0, ("Lookup Failed for: " + nme).c_str(), "Lookup Failed", MB_ICONERROR);
        }
      }
    }
    WSACleanup();
  }

  if (bDoRedirectAgent) {
    // Redirecting agent
    using namespace nsDetourConnect;

    // MessageBox(0, ("Redirecting agent to " + defaultAgentIP + " " + std::to_string(config->hyperbotPort)).c_str(), "Redirect", MB_OK);

    std::vector<std::string> tokens = sro::edx_labs::TokenizeString(defaultAgentIP, " .");

    if (tokens.size() != 4) {
      MessageBox(0, "Please enter an IP in the format 0.0.0.0", "Fatal Error", MB_ICONERROR);
      return;
    }

    routeArray[routeListCount].dstA = atoi(tokens[0].c_str());
    routeArray[routeListCount].dstB = atoi(tokens[1].c_str());
    routeArray[routeListCount].dstC = atoi(tokens[2].c_str());
    routeArray[routeListCount].dstD = atoi(tokens[3].c_str());
    routeArray[routeListCount].dstPort = config->hyperbotPort;

    routeArray[routeListCount].srcA = 255;
    routeArray[routeListCount].srcB = 255;
    routeArray[routeListCount].srcC = 255;
    routeArray[routeListCount].srcD = 255;

    routeArray[routeListCount].srcPort = config->agentPort; // rather than doing another GATEPORT access

    routeListCount++;
  }
}

//-------------------------------------------------------------------------

// The function where we place all our logic
void UserOnInject(sro::edx_labs::DllConfig *config) {
  char moduleName[MAX_PATH + 1] = { 0 };
  GetModuleFileName(0, moduleName, MAX_PATH);

  std::string mpk2 = moduleName;
  mpk2 = mpk2.substr(0, 1 + mpk2.find_last_of("\\/"));
  mpk2 += "media.pk2";

  try {
    sro::pk2::Pk2ReaderModern pk2Reader(mpk2);
    auto divisionInfoEntry = pk2Reader.getEntry("DIVISIONINFO.TXT");
    auto divisionInfoData = pk2Reader.getEntryData(divisionInfoEntry);
    gDivisionInfo = sro::pk2::parsing::parseDivisionInfo(divisionInfoData);
  } catch (std::exception &ex) {
    std::string errorMsg = ex.what();
    errorMsg = "The DivisionInfo could not be parsed.\nError: \"" + errorMsg + "\"";
    MessageBox(0, errorMsg.c_str(), "Fatal Error", MB_ICONERROR);
  }

  // If we obtained the locale from the PK2
  if (gDivisionInfo.locale != 0)
  {
    switch (gDivisionInfo.locale)
    {
    case 2: languageFlag = 0; break; // KSRO
    case 4: languageFlag = 1; break; // CSRO
    case 12: languageFlag = 2; break; // TSRO
    case 18: languageFlag = 4; break; // ISRO
    case 23: languageFlag = 5; break; // VSRO
    case 15: languageFlag = 3; break; // JSRO - thanks kaperucito!
    default: /*MessageBox(0, "An unsupported locale was detected.", "Fatal Error", MB_ICONERROR);*/ bDoLanguagePatch = false; // uh-oh?
    }
  }

  // Create the configuration dialog
  modifyRoutelist(config);
  /*HWND hWnd = CreateDialogParam(gInstance, MAKEINTRESOURCE(IDD_DIALOG1), NULL, Main_DlgProc, (LPARAM)gInstance);
  MSG msg;
  while (GetMessage(&msg, NULL, 0, 0))
  {
    if (IsDialogMessage(hWnd, &msg) == 0)
    {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
  }
  DestroyWindow(hWnd);*/

  /*if (bDebugConsole)
  {
    // Create a debugging console
    CreateConsole("edxSilkroadLoader_Lite Debugging Console");
  }*/

  DetourRestoreAfterWith();
  DetourTransactionBegin();
  DetourUpdateThread(GetCurrentThread());
  DetourAttach(&(PVOID&)nsDetourLoadStringA::Real_LoadString, nsDetourLoadStringA::Detour_LoadString);
  DetourTransactionCommit();
}

//-------------------------------------------------------------------------

void UserOnUninject()
{
  if (bDoRedirectGateway || bDoRedirectAgent)
  {
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourDetach(&(PVOID&)nsDetourConnect::Real_connect, nsDetourConnect::Detour_connect);
    DetourTransactionCommit();
  }

  if (bDebugConsole)
  {
    FreeConsole();
  }
}

//-------------------------------------------------------------------------

void OnConsoleInput(std::string input)
{
  if (input == "/min")
  {
    ShowWindow(GetActiveWindow(), SW_MINIMIZE);
  }
  else if (input == "/exit")
  {
    ExitProcess(0);
  }
}

//-------------------------------------------------------------------------

DWORD WINAPI InjectionThread(LPVOID lpParam)
{
  HINSTANCE exeInstance = (HINSTANCE)lpParam;

  CreateMutexA(0, 0, "Silkroad Online Launcher");
  CreateMutexA(0, 0, "Ready");

  BYTE peHeader[4096];
  sro::edx_labs::ReadBytes(PtrToUlong(exeInstance), peHeader, 4096);
  PIMAGE_DOS_HEADER dosHeader = (PIMAGE_DOS_HEADER)peHeader;
#define MakePtr(cast, ptr, addValue) (cast)((DWORD)(ptr)+(DWORD)(addValue))
  PIMAGE_NT_HEADERS pNTHeader = MakePtr(PIMAGE_NT_HEADERS, dosHeader, dosHeader->e_lfanew);
#undef MakePtr

  DWORD codeStart = 0;
  DWORD codeSize = 0;
  LPBYTE codePtr = 0;

  DWORD dataStart = 0;
  DWORD dataSize = 0;
  LPBYTE dataPtr = 0;

  IMAGE_SECTION_HEADER * pish = (IMAGE_SECTION_HEADER *)(((LPBYTE)&pNTHeader->OptionalHeader) + pNTHeader->FileHeader.SizeOfOptionalHeader);
  for (int x = 0; x < pNTHeader->FileHeader.NumberOfSections; ++x)
  {
    IMAGE_SECTION_HEADER & header = pish[x];

    if (header.VirtualAddress == pNTHeader->OptionalHeader.BaseOfData)
    {
      printf("-- Data --\n");
      dataStart = pNTHeader->OptionalHeader.ImageBase + header.VirtualAddress;
      dataSize = header.Misc.VirtualSize;
      printf("\tdataStart: %X\n", dataStart);
      printf("\tdataSize: %X\n", dataSize);
    }
    else if (header.VirtualAddress == pNTHeader->OptionalHeader.BaseOfCode)
    {
      printf("-- Code --\n");
      codeStart = pNTHeader->OptionalHeader.ImageBase + header.VirtualAddress;
      codeSize = header.Misc.VirtualSize;
      printf("\tdataStart: %X\n", codeStart);
      printf("\tdataSize: %X\n", codeSize);
    }
    else
    {
      continue;
    }
  }

  if (codeSize == 0)
  {
    MessageBoxA(0, "codeSize == 0", "Fatal Error", MB_ICONERROR);
    ExitProcess(0);
  }
  if (dataSize == 0)
  {
    MessageBoxA(0, "dataSize == 0", "Fatal Error", MB_ICONERROR);
    ExitProcess(0);
  }

  codePtr = (LPBYTE)malloc(codeSize);
  sro::edx_labs::ReadBytes(codeStart, codePtr, codeSize);

  dataPtr = (LPBYTE)malloc(dataSize);
  sro::edx_labs::ReadBytes(dataStart, dataPtr, dataSize);

  std::vector<LONGLONG> results;

  // Security seed fix
  if (bDoSecuritySeed)
  {
    do
    {
      BYTE securitySeedSig[] =
      {
        0x8B, 0x4C, 0x24, 0x04, 0x81, 0xE1, 0xFF, 0xFF,
        0xFF, 0x7F
      };
      results = sro::edx_labs::FindSignature(securitySeedSig, 0, sizeof(securitySeedSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }

      DWORD secSeedAddr = (DWORD)(results[0] + codeStart);
      printf("secSeedAddr: 0x%X\n", secSeedAddr);

      BYTE patch1[] = { 0xB9, 0x33, 0x00, 0x00, 0x00, 0x90, 0x90, 0x90, 0x90, 0x90 };
      sro::edx_labs::WriteBytes(secSeedAddr, patch1, sizeof(patch1));

      printf("\n");
    } while (false);
  }

  // Proxy redirect
  if (bDoRedirectGateway || bDoRedirectAgent)
  {
    // Detour API functions
    DetourRestoreAfterWith();
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach(&(PVOID&)nsDetourConnect::Real_connect, nsDetourConnect::Detour_connect);
    DetourTransactionCommit();
  }

  // Swear Filter
  if (bDoSwearFilter)
  {
    do
    {
      // Part 1 - Find the string in the client.
      // UNICODE "UIIT_MSG_CHATWND_MESSAGE_FILTER"

      BYTE abuseFilterStringSig[] =
      {
        0x55, 0x00, 0x49, 0x00, 0x49, 0x00, 0x54, 0x00,
        0x5F, 0x00, 0x4D, 0x00, 0x53, 0x00, 0x47, 0x00,
        0x5F, 0x00, 0x43, 0x00, 0x48, 0x00, 0x41, 0x00,
        0x54, 0x00, 0x57, 0x00, 0x4E, 0x00, 0x44, 0x00,
        0x5F, 0x00, 0x4D, 0x00, 0x45, 0x00, 0x53, 0x00,
        0x53, 0x00, 0x41, 0x00, 0x47, 0x00, 0x45, 0x00,
        0x5F, 0x00, 0x46, 0x00, 0x49, 0x00, 0x4C, 0x00,
        0x54, 0x00, 0x45, 0x00, 0x52, 0x00, 0x00, 0x00
      };
      results = sro::edx_labs::FindSignature(abuseFilterStringSig, 0, sizeof(abuseFilterStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }

      DWORD logicalAddress1 = (DWORD)(results[0] + dataStart);
      printf("logicalAddress1: 0x%X\n", logicalAddress1);

      // Part 2 - Perform the signature search based on the client's address

      BYTE abuseFilterSig[] =
      {
        0x68, 0x00, 0x00, 0x00, 0x00
      };
      memcpy(abuseFilterSig + 1, &logicalAddress1, 4);

      results = sro::edx_labs::FindSignature(abuseFilterSig, 0, sizeof(abuseFilterSig), codePtr, codeSize);
      if (results.size() <= 3)
      {
        printf("%i results were returned. 4 or more were expected. Please use an updated signature.\n", results.size());
        break;
      }

      for (size_t x = 0; x < 3; ++x)
      {
        DWORD logicalAddress2 = (DWORD)(results[x] + codeStart);
        DWORD patchAddress = logicalAddress2 - 2;
        printf("patchAddress: 0x%X\n", patchAddress);

        BYTE patch1[] = { 0xEB };
        sro::edx_labs::WriteBytes(patchAddress, patch1, sizeof(patch1));
      }

      printf("\n");
    } while (false);
  }

  // Language filter patches
  if (bDoLanguagePatch)
  {
    do
    {
      BYTE koreanLanguageStringSig[] = { 0x4B, 0x6F, 0x72, 0x65, 0x61, 0x6E, 0x00 };
      BYTE chineseLanguageStringSig[] = { 0x43, 0x68, 0x69, 0x6E, 0x65, 0x73, 0x65, 0x00 };
      BYTE taiwanLanguageStringSig[] = { 0x54, 0x61, 0x69, 0x77, 0x61, 0x6E, 0x00 };
      BYTE japanLanguageStringSig[] = { 0x4A, 0x61, 0x70, 0x61, 0x6E, 0x00 };
      BYTE englishLanguageStringSig[] = { 0x45, 0x6E, 0x67, 0x6C, 0x69, 0x73, 0x68, 0x00 };
      BYTE vietnamLanguageStringSig[] = { 0x56, 0x69, 0x65, 0x74, 0x6E, 0x61, 0x6D, 0x00 };

      DWORD physicalKoreanStringAddress = 0;
      DWORD physicalChineseStringAddress = 0;
      DWORD physicalTaiwanStringAddress = 0;
      DWORD physicalJapanStringAddress = 0;
      DWORD physicalEnglishStringAddress = 0;
      DWORD physicalVietnamStringAddress = 0;

      DWORD logicalKoreanStringAddress = 0;
      DWORD logicalChineseStringAddress = 0;
      DWORD logicalTaiwanStringAddress = 0;
      DWORD logicalJapanStringAddress = 0;
      DWORD logicalEnglishStringAddress = 0;
      DWORD logicalVietnamStringAddress = 0;

      DWORD physicalCharSelectStringAddress = 0;
      DWORD logicalCharSelectStringAddress = 0;

      DWORD physicalLauncherStringAddress = 0;
      DWORD logicalLauncherStringAddress = 0;

      results = sro::edx_labs::FindSignature(koreanLanguageStringSig, 0, sizeof(koreanLanguageStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      physicalKoreanStringAddress = (DWORD)(results[0] + dataStart);

      results = sro::edx_labs::FindSignature(chineseLanguageStringSig, 0, sizeof(chineseLanguageStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      physicalChineseStringAddress = (DWORD)(results[0] + dataStart);

      results = sro::edx_labs::FindSignature(taiwanLanguageStringSig, 0, sizeof(taiwanLanguageStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      physicalTaiwanStringAddress = (DWORD)(results[0] + dataStart);

      results = sro::edx_labs::FindSignature(japanLanguageStringSig, 0, sizeof(japanLanguageStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      physicalJapanStringAddress = (DWORD)(results[0] + dataStart);

      results = sro::edx_labs::FindSignature(englishLanguageStringSig, 0, sizeof(englishLanguageStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      physicalEnglishStringAddress = (DWORD)(results[0] + dataStart);

      results = sro::edx_labs::FindSignature(vietnamLanguageStringSig, 0, sizeof(vietnamLanguageStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      physicalVietnamStringAddress = (DWORD)(results[0] + dataStart);

      printf("physicalKoreanStringAddress: 0x%X\n", physicalKoreanStringAddress);
      printf("physicalChineseStringAddress: 0x%X\n", physicalChineseStringAddress);
      printf("physicalTaiwanStringAddress: 0x%X\n", physicalTaiwanStringAddress);
      printf("physicalJapanStringAddress: 0x%X\n", physicalJapanStringAddress);
      printf("physicalEnglishStringAddress: 0x%X\n", physicalEnglishStringAddress);
      printf("physicalVietnamStringAddress: 0x%X\n", physicalVietnamStringAddress);
      printf("\n");

      BYTE languageStringLoadSig[] = { 0xBE, 0x00, 0x00, 0x00, 0x00 };

      memcpy(languageStringLoadSig + 1, &physicalKoreanStringAddress, 4);
      results = sro::edx_labs::FindSignature(languageStringLoadSig, 0, sizeof(languageStringLoadSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      logicalKoreanStringAddress = (DWORD)(results[0] + codeStart);

      memcpy(languageStringLoadSig + 1, &physicalChineseStringAddress, 4);
      results = sro::edx_labs::FindSignature(languageStringLoadSig, 0, sizeof(languageStringLoadSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      logicalChineseStringAddress = (DWORD)(results[0] + codeStart);

      memcpy(languageStringLoadSig + 1, &physicalTaiwanStringAddress, 4);
      results = sro::edx_labs::FindSignature(languageStringLoadSig, 0, sizeof(languageStringLoadSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      logicalTaiwanStringAddress = (DWORD)(results[0] + codeStart);

      memcpy(languageStringLoadSig + 1, &physicalJapanStringAddress, 4);
      results = sro::edx_labs::FindSignature(languageStringLoadSig, 0, sizeof(languageStringLoadSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      logicalJapanStringAddress = (DWORD)(results[0] + codeStart);

      memcpy(languageStringLoadSig + 1, &physicalEnglishStringAddress, 4);
      results = sro::edx_labs::FindSignature(languageStringLoadSig, 0, sizeof(languageStringLoadSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      logicalEnglishStringAddress = (DWORD)(results[0] + codeStart);

      memcpy(languageStringLoadSig + 1, &physicalVietnamStringAddress, 4);
      results = sro::edx_labs::FindSignature(languageStringLoadSig, 0, sizeof(languageStringLoadSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      logicalVietnamStringAddress = (DWORD)(results[0] + codeStart);

      // Calculate the positions of the JNZs for each specific language
      // (we can't really automate this since the JNZ byte of 0x75 is valid in an address)

      logicalKoreanStringAddress += 0xC;
      printf("logicalKoreanStringAddress: 0x%X\n", logicalKoreanStringAddress);

      logicalChineseStringAddress += 0xE;
      printf("logicalChineseStringAddress: 0x%X\n", logicalChineseStringAddress);

      logicalTaiwanStringAddress += 0xE;
      printf("logicalTaiwanStringAddress: 0x%X\n", logicalTaiwanStringAddress);

      logicalJapanStringAddress += 0xE;
      printf("logicalJapanStringAddress: 0x%X\n", logicalJapanStringAddress);

      logicalEnglishStringAddress += 0xE;
      printf("logicalEnglishStringAddress: 0x%X\n", logicalEnglishStringAddress);

      logicalVietnamStringAddress += 0xE;
      printf("logicalVietnamStringAddress: 0x%X\n", logicalVietnamStringAddress);

      printf("\n");

      BYTE patch1[] = { 0xEB };
      sro::edx_labs::WriteBytes(logicalKoreanStringAddress, patch1, sizeof(patch1));
      sro::edx_labs::WriteBytes(logicalChineseStringAddress, patch1, sizeof(patch1));
      sro::edx_labs::WriteBytes(logicalTaiwanStringAddress, patch1, sizeof(patch1));
      sro::edx_labs::WriteBytes(logicalJapanStringAddress, patch1, sizeof(patch1));
      sro::edx_labs::WriteBytes(logicalVietnamStringAddress, patch1, sizeof(patch1));

      BYTE patch2[] = { 0x90, 0x90 };
      sro::edx_labs::WriteBytes(logicalEnglishStringAddress, patch2, sizeof(patch2));

      BYTE charSelectStringSig[] = { 0x43, 0x68, 0x61, 0x72, 0x53, 0x65, 0x6C, 0x65, 0x63, 0x74, 0x00 };

      results = sro::edx_labs::FindSignature(charSelectStringSig, 0, sizeof(charSelectStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      physicalCharSelectStringAddress = (DWORD)(results[0] + dataStart);

      printf("physicalCharSelectStringAddress: 0x%X\n", physicalCharSelectStringAddress);

      BYTE charSelectStringLoadSig[] = { 0x68, 0x00, 0x00, 0x00, 0x00 };
      memcpy(charSelectStringLoadSig + 1, &physicalCharSelectStringAddress, 4);
      results = sro::edx_labs::FindSignature(charSelectStringLoadSig, 0, sizeof(charSelectStringLoadSig), codePtr, codeSize);
      if (results.size() != 2)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 2);
        break;
      }
      logicalCharSelectStringAddress = (DWORD)(results[1] + codeStart);

      printf("logicalCharSelectStringAddress: 0x%X\n", logicalCharSelectStringAddress);

      DWORD callOffset = *(LPDWORD)(codePtr + results[1] + 0x15 + 1);
      DWORD callAddr = logicalCharSelectStringAddress + 0x15 + callOffset + 5;

      printf("callOffset: 0x%X\n", callOffset);
      printf("callAddr: 0x%X\n", callAddr);

      nsEnglishPatch::GetGlobalSettingsFunc = (FARPROC)(callAddr);

      printf("\n");

      BYTE launcherStringSig[] = { 0x50, 0x6C, 0x65, 0x61, 0x73, 0x65, 0x20, 0x45, 0x78, 0x65, 0x63, 0x75, 0x74, 0x65, 0x20, 0x74, 0x68, 0x65, 0x20, 0x22, 0x53, 0x69, 0x6C, 0x6B, 0x72, 0x6F, 0x61, 0x64, 0x2E, 0x65, 0x78, 0x65, 0x2E, 0x22, 0x00 };

      results = sro::edx_labs::FindSignature(launcherStringSig, 0, sizeof(launcherStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      physicalLauncherStringAddress = (DWORD)(results[0] + dataStart);

      printf("physicalLauncherStringAddress: 0x%X\n", physicalLauncherStringAddress);

      BYTE launcherStringLoadSig[] = { 0x68, 0x00, 0x00, 0x00, 0x00 };
      memcpy(launcherStringLoadSig + 1, &physicalLauncherStringAddress, 4);
      results = sro::edx_labs::FindSignature(launcherStringLoadSig, 0, sizeof(launcherStringLoadSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      logicalLauncherStringAddress = (DWORD)(results[0] + codeStart);

      printf("logicalLauncherStringAddress: 0x%X\n", logicalLauncherStringAddress);

      DWORD codecaveAddr = logicalLauncherStringAddress + 0x6C;
      nsEnglishPatch::customLangAddr = *(LPDWORD)(codePtr + results[0] + 0x6C + 2);

      printf("codecaveAddr: 0x%X\n", codecaveAddr);
      printf("customMultiAddr: 0x%X\n", nsEnglishPatch::customLangAddr);

      sro::edx_labs::CreateCodeCave(codecaveAddr, 7, nsEnglishPatch::codecave_EnglishPatch);

      printf("\n");
    } while (false);
  }

  // Nude patch
  if (bDoNudePatch)
  {
    do
    {
      BYTE nudePatchSig[] =
      {
        0x8B, 0x84, 0xEE, 0x1C, 0x01, 0x00, 0x00, 0x3B,
        0x44, 0x24, 0x14
      };
      results = sro::edx_labs::FindSignature(nudePatchSig, 0, sizeof(nudePatchSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }

      DWORD nudePatchAddr = (DWORD)(results[0] + sizeof(nudePatchSig) + codeStart);
      printf("nudePatchAddr: 0x%X\n", nudePatchAddr);

      BYTE patch1[] = { 0x90, 0x90 };
      sro::edx_labs::WriteBytes(nudePatchAddr, patch1, sizeof(patch1));

      printf("\n");
    } while (false);
  }

  // Zoom hack
  if (bDoZoomhack)
  {
    do
    {
      BYTE zoomHackSig[] =
      {
        0xDF, 0xE0, 0xF6, 0xC4, 0x41, 0x7A, 0x08, 0xD9,
        0x9E
      };
      results = sro::edx_labs::FindSignature(zoomHackSig, 0, sizeof(zoomHackSig), codePtr, codeSize);
      if (results.size() != 2)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 2);
        break;
      }

      DWORD zoomHackAddr = (DWORD)(results[1] + sizeof(zoomHackSig) - 4 + codeStart);
      printf("zoomHackAddr: 0x%X\n", zoomHackAddr);

      BYTE patch1[] = { 0xEB };
      sro::edx_labs::WriteBytes(zoomHackAddr, patch1, sizeof(patch1));

      printf("\n");
    } while (false);
  }

  // Multiclient
  if (bDoMulticlient)
  {
    do
    {
      // Part 1 - Find the string in the client.
      // UNICODE "UIIT_MSG_CHATWND_MESSAGE_FILTER"

      BYTE mutexStringSig[] =
      {
        0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F,
        0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F,
        0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F,
        0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F,
        0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F,
        0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F,
        0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F,
        0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F, 0x2F,
        0x2F, 0x2F, 0x00
      };
      results = sro::edx_labs::FindSignature(mutexStringSig, 0, sizeof(mutexStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }

      DWORD mutexStringAddress = (DWORD)(results[0] + dataStart);
      printf("mutexStringAddress: 0x%X\n", mutexStringAddress);

      BYTE mutexSig[] =
      {
        0x68, 0x00, 0x00, 0x00, 0x00
      };
      memcpy(mutexSig + 1, &mutexStringAddress, 4);

      results = sro::edx_labs::FindSignature(mutexSig, 0, sizeof(mutexSig), codePtr, codeSize);
      if (results.size() != 2)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 2);
        break;
      }

      DWORD logicalAddress2 = (DWORD)(results[0] + codeStart);
      DWORD patchAddress = logicalAddress2 - 2;
      printf("patchAddress: 0x%X\n", patchAddress);

      BYTE patch1[] = { 0xEB };
      sro::edx_labs::WriteBytes(patchAddress, patch1, sizeof(patch1));

      BYTE macAddressSig[] = { 0x6A, 0x06, 0x8D, 0x44, 0x24, 0x48, 0x50, 0x8B, 0xCF };

      results = sro::edx_labs::FindSignature(macAddressSig, 0, sizeof(macAddressSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      DWORD macAddrSigAddr = (DWORD)(results[0] + codeStart);

      printf("macAddrSigAddr: 0x%X\n", macAddrSigAddr);

      DWORD codecaveAddr = macAddrSigAddr + sizeof(macAddressSig);
      DWORD callOffset = *(LPDWORD)(codePtr + results[0] + sizeof(macAddressSig) + 1);
      DWORD callAddr = macAddrSigAddr + sizeof(macAddressSig) + callOffset + 5;
      nsMulticlient::AppendBytesFunc = (FARPROC)(callAddr);

      printf("codecaveAddr: 0x%X\n", codecaveAddr);
      printf("callOffset: 0x%X\n", callOffset);
      printf("callAddr: 0x%X\n", callAddr);

      sro::edx_labs::CreateCodeCave(codecaveAddr, 5, nsMulticlient::codecave_Multiclient);

      BYTE bindSig[] = { 0x74, 0x3D, 0x68, 0xA3, 0x3D, 0x00, 0x00 };

      results = sro::edx_labs::FindSignature(bindSig, 0, sizeof(bindSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }
      DWORD bindSigAddr = (DWORD)(results[0] - 0x51 + codeStart);

      printf("bindSigAddr: 0x%X\n", bindSigAddr);

      BYTE patch2[] = { 0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3 };
      sro::edx_labs::WriteBytes(bindSigAddr, patch2, sizeof(patch2));

      printf("\n");
    } while (false);
  }

  if (bHookInput)
  {
    do
    {
      BYTE chattingStringSig[] =
      {
        0x55, 0x00, 0x49, 0x00, 0x49, 0x00, 0x54, 0x00,
        0x5F, 0x00, 0x4D, 0x00, 0x53, 0x00, 0x47, 0x00,
        0x5F, 0x00, 0x43, 0x00, 0x41, 0x00, 0x4E, 0x00,
        0x54, 0x00, 0x5F, 0x00, 0x43, 0x00, 0x48, 0x00,
        0x41, 0x00, 0x54, 0x00, 0x54, 0x00, 0x49, 0x00,
        0x4E, 0x00, 0x47, 0x00, 0x00, 0x00
      };
      results = sro::edx_labs::FindSignature(chattingStringSig, 0, sizeof(chattingStringSig), dataPtr, dataSize);
      if (results.size() != 1)
      {
        printf("[%s] %i results were returned. Only %i were expected. Please use an updated signature.\n", "chattingStringSig", results.size(), 1);
        break;
      }

      DWORD chattingStringPhysicalAddress = (DWORD)(results[0] + dataStart);
      printf("chattingStringPhysicalAddress: 0x%X\n", chattingStringPhysicalAddress);

      BYTE chattingSig[] =
      {
        0x68, 0x00, 0x00, 0x00, 0x00
      };
      memcpy(chattingSig + 1, &chattingStringPhysicalAddress, 4);

      results = sro::edx_labs::FindSignature(chattingSig, 0, sizeof(chattingSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("[%s] %i results were returned. Only %i were expected. Please use an updated signature.\n", "chattingSig", results.size(), 1);
        break;
      }

      DWORD chattingLogicalAddress = (DWORD)(results[0] + codeStart);
      printf("chattingLogicalAddress: 0x%X\n", chattingLogicalAddress);

      DWORD newSearchAddrOffset = chattingLogicalAddress + 0x200 - codeStart;

      BYTE patchSig[] =
      {
        0x89, 0x6C, 0x24, 0x00,
        0x89, 0x6C, 0x24, 0x00,
        0x89, 0x6C, 0x24, 0x00,
        0x89, 0x6C, 0x24, 0x00
      };
      BYTE patchSigWildcard[] =
      {
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x01
      };

      results = sro::edx_labs::FindSignature(patchSig, patchSigWildcard, sizeof(patchSig), codePtr + newSearchAddrOffset, 0x200);
      if (results.size() != 1)
      {
        printf("[%s] %i results were returned. Only %i were expected. Please use an updated signature.\n", "patchSig", results.size(), 1);
        break;
      }

      DWORD customAddr = *(LPDWORD)(codePtr + newSearchAddrOffset + results[0] + sizeof(patchSig) + 2);
      printf("customAddr: 0x%X\n", customAddr);
      nsHookInput::hookInputCustomAddr = customAddr;

      DWORD patchLogicalAddress = (DWORD)(newSearchAddrOffset + results[0] + codeStart + sizeof(patchSig));
      printf("patchLogicalAddress: 0x%X\n", patchLogicalAddress);
      sro::edx_labs::CreateCodeCave(patchLogicalAddress, 6, nsHookInput::codecave_HookInput);

      printf("\n");
    } while (false);
  }

  if (bPatchKsroImgCode)
  {
    do
    {
      BYTE imgCodeSig[] =
      {
        0x66, 0xC7, 0x00, 0x23, 0x63
      };
      results = sro::edx_labs::FindSignature(imgCodeSig, 0, sizeof(imgCodeSig), codePtr, codeSize);
      if (results.size() != 1)
      {
        printf("%i results were returned. Only %i were expected. Please use an updated signature.\n", results.size(), 1);
        break;
      }

      DWORD ImgCodeCaveAddr = (DWORD)(results[0] + codeStart + 0x1A);
      printf("ImgCodeCaveAddr: 0x%X\n", ImgCodeCaveAddr);

      DWORD callOffset = *(LPDWORD)(codePtr + results[0] + 0x1A + 1);
      DWORD callAddr = ImgCodeCaveAddr + callOffset + 5;
      printf("callOffset: 0x%X\n", callOffset);
      printf("callAddr: 0x%X\n", callAddr);

      nsEnglishCaptcha::AppendStringFunc = (FARPROC)callAddr;

      sro::edx_labs::CreateCodeCave(ImgCodeCaveAddr, 5, nsEnglishCaptcha::codecave_EnglishCaptcha);

      printf("\n");
    } while (false);
  }

  free(codePtr);
  free(dataPtr);

  return 0;
}

//-------------------------------------------------------------------------