#ifndef SRO_SHARED_IO_H_
#define SRO_SHARED_IO_H_

#include <cstdint>
#include <cstdio>

namespace sro::shared_io {

int file_seek(std::FILE *file, int64_t offset, int orgin);

int64_t file_tell(std::FILE *file);

int file_remove(const char *filename);

// std::vector<uint8_t> file_tovector(const char * filename);

} // namespace sro::shared_io

#endif // SRO_SHARED_IO_H_