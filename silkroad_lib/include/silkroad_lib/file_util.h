#ifndef SRO_FILE_UTIL_H_
#define SRO_FILE_UTIL_H_
#include <filesystem>

namespace sro::file_util {

std::filesystem::path getAppDataPath();

} // namespace sro::file_util

#endif // SRO_FILE_UTIL_H_