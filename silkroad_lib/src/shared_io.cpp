#include "shared_io.hpp"

#include <cstdlib>

#if defined(_WIN32)
	#include <windows.h>
#else
	#include <dirent.h>
#endif

namespace sro::shared_io {

int file_seek(std::FILE *file, int64_t offset, int orgin) {
#if defined(_WIN32)
	return _fseeki64(file, offset, orgin);
#else
	return std::fseek(file, offset, orgin);
#endif
}

int64_t file_tell(std::FILE * file) {
#if defined(_WIN32)
	return _ftelli64(file);
#else
	return std::ftell(file);
#endif
}

int file_remove(const char * filename) {
#if defined(_WIN32)
	return DeleteFileA(filename);
#else
	return std::remove(filename);
#endif
}

// std::vector<uint8_t> file_tovector(const char * filename) {
// 	std::vector<uint8_t> contents;
// 	std::FILE * infile = 0;
// #if defined(_WIN32)
// 	fopen_s(&infile, filename, "rb");
// #else
// 	infile = fopen(filename, "rb");
// #endif
// 	if(infile == 0) {
// 		return contents;
// 	}
// 	file_seek(infile, 0, SEEK_END);
// 	int64_t size = file_tell(infile);
// 	file_seek(infile, 0, SEEK_SET);
// 	contents.resize(size);
// 	size_t read_count = 0;
// 	int64_t index = 0;
// 	while(size && (read_count = fread(&contents[index], 1, size > 0x7FFFFFF ? 0x7FFFFFF : size, infile))) {
// 		index += read_count;
// 		size -= read_count;
// 	}
// 	fclose(infile);
// 	return contents;
// }

} // namespace sro::shared_io