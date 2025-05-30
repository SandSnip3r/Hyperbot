#ifndef DLL_CONFIG_HPP_
#define DLL_CONFIG_HPP_

#include <cstdint>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace sro::edx_labs {

#pragma pack(push, 1)
struct DllConfig {
  DWORD size;
  WORD gatewayPort;
  WORD agentPort;
  WORD hyperbotPort;
};
#pragma pack(pop)


} // namespace sro::edx_labs

#endif // DLL_CONFIG_HPP_