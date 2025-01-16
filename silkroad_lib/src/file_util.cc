#include "file_util.h"

namespace sro::file_util {

std::filesystem::path getAppDataPath() {
  std::filesystem::path appDataPath;
#if defined(_WIN32)
  PWSTR pathTmp;

  /* Attempt to get user's AppData folder
  *
  * Microsoft Docs:
  * https://docs.microsoft.com/en-us/windows/win32/api/shlobj_core/nf-shlobj_core-shgetknownfolderpath
  * https://docs.microsoft.com/en-us/windows/win32/shell/knownfolderid
  */
  auto getFolderPathResult = SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &pathTmp);

  // Error check
  if (getFolderPathResult != S_OK) {
    CoTaskMemFree(pathTmp);
    return {};
  }
  // Convert the Windows path type to a C++ path
  appDataPath = pathTmp;
  // Free memory
  CoTaskMemFree(pathTmp);
#else
  const char* xdgConfigHome = std::getenv("XDG_CONFIG_HOME");
  if (xdgConfigHome && *xdgConfigHome) {
    appDataPath = xdgConfigHome;
  } else {
    const char* home = std::getenv("HOME");
    if (home && *home) {
      appDataPath = std::filesystem::path(home) / ".config";
    } else {
      throw std::runtime_error("Failed to retrieve AppData path on Linux: neither XDG_CONFIG_HOME nor HOME is set.");
    }
  }
#endif

  const std::string kAppDataSubdirName = "Hyperbot";
  return {appDataPath / kAppDataSubdirName};
}

} // namespace sro::file_util