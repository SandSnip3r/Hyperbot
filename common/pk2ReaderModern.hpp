#ifndef PK2_READER_MODERN_HPP
#define PK2_READER_MODERN_HPP

#include "PK2.h"
#include "PK2Reader.h"

#include <filesystem>
#include <string>
#include <vector>

namespace pk2 {

class Pk2ReaderModern {
public:
	Pk2ReaderModern(const std::experimental::filesystem::v1::path &pk2Path);
	~Pk2ReaderModern();
	PK2Entry getEntry(const std::string &entryName);
	std::vector<uint8_t> getEntryData(PK2Entry &entry);
private:
	std::experimental::filesystem::v1::path pk2Path_;
	PK2Reader pk2Reader_;
};

} // namespace pk2

#endif // PK2_READER_MODERN_HPP