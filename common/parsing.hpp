#ifndef PK2_PARSING_HPP_
#define PK2_PARSING_HPP_

#include "divisionInfo.hpp"
#include "itemInfo.hpp"
#include "skill.hpp"

#include <filesystem>
#include <ostream>
#include <vector>

namespace pk2::parsing {

// Transforms raw pk2 data into a std::string
//
// [param] data   Data from a PK2Entry
// [return]       data as an std::string
std::string fileDataToString(const std::vector<uint8_t> &data);

// Parses string representing a line of text from skilldata_xxxx.txt in the Media.pk2 into a Skill object
//
// [param] line   A line of text
// [return]       A populated Skill object
pk2::media::Skill parseSkilldataLine(const std::string &line);

// Parses string representing a line of text from itemdata_xxxx.txt in the Media.pk2 into an Item object
//
// [param] line   A line of text
// [return]       A populated Item object
pk2::media::Item parseItemdataLine(const std::string &line);

// Parses raw pk2 data into DivisionInfo object
//
// [param] data   Data from the DIVISIONINFO.txt PK2Entry
// [return]       Populated DivisionInfo object
DivisionInfo parseDivisionInfo(const std::vector<uint8_t> &data);

// Splits a string into pieces
//
// [param] str    string to split
// [param] delim  delimiter to split the string `str` on
// [return]       string split into pieces
std::vector<std::string> split(const std::string &str, const std::string &delim);

// Splits a string into pieces and returns only the desired pieces
//
// [param] str    string to split
// [param] delim  delimiter to split the string `str` on
// [param] fields which fields to keep
// [return]       Populated DivisionInfo object
std::vector<std::string> splitAndSelectFields(const std::string &str, const std::string &delim, const std::vector<int> &fields);

template<typename T>
T get(const std::vector<uint8_t> &data, int &readIndex) {
	if (readIndex + sizeof(T) > data.size()) {
		throw std::runtime_error("Trying to get data from past-end");
	}
	T result{0};
	for (int i=0; i<sizeof(T); ++i) {
		result <<= 8;
		result |= data[readIndex+sizeof(T)-i-1];
	}
	readIndex += sizeof(T);
	return result;
}

} // namespace pk2::parsing

#endif // PK2_PARSING_HPP_