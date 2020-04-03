#ifndef PK2_PARSING_HPP_
#define PK2_PARSING_HPP_

#include "../divisionInfo.hpp"
#include "../ref/character.hpp"
#include "../ref/item.hpp"
#include "../ref/mappingShopGroup.hpp"
#include "../ref/mappingShopWithTab.hpp"
#include "../ref/scrapOfPackageItem.hpp"
#include "../ref/shopGood.hpp"
#include "../ref/shopGroup.hpp"
#include "../ref/shopTab.hpp"
#include "../ref/skill.hpp"
#include "../ref/teleport.hpp"

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
pk2::ref::Skill parseSkilldataLine(const std::string &line);

// Parses string representing a line of text from characterdata_xxxx.txt in the Media.pk2 into an Character object
//
// [param] line   A line of text
// [return]       A populated Character object
pk2::ref::Character parseCharacterdataLine(const std::string &line);

// Parses string representing a line of text from itemdata_xxxx.txt in the Media.pk2 into an Item object
//
// [param] line   A line of text
// [return]       A populated Item object
pk2::ref::Item parseItemdataLine(const std::string &line);

// Parses string representing a line of text from teleportbuilding.txt in the Media.pk2 into a Teleport object
//
// [param] line   A line of text
// [return]       A populated Teleport object
pk2::ref::Teleport parseTeleportbuildingLine(const std::string &line);

// Parses string representing a line of text from refscrapofpackageitem.txt in the Media.pk2 into a ScrapOfPackageItem object
//
// [param] line   A line of text
// [return]       A populated ScrapOfPackageItem object
pk2::ref::ScrapOfPackageItem parseScrapOfPackageItemLine(const std::string &line);

// Parses string representing a line of text from refshoptab.txt in the Media.pk2 into a ShopTab object
//
// [param] line   A line of text
// [return]       A populated ShopTab object
pk2::ref::ShopTab parseShopTabLine(const std::string &line);

// Parses string representing a line of text from refshopgroup.txt in the Media.pk2 into a ShopGroup object
//
// [param] line   A line of text
// [return]       A populated ShopGroup object
pk2::ref::ShopGroup parseShopGroupLine(const std::string &line);

// Parses string representing a line of text from refshopgoods.txt in the Media.pk2 into a ShopGood object
//
// [param] line   A line of text
// [return]       A populated ShopGood object
pk2::ref::ShopGood parseShopGoodLine(const std::string &line);

// Parses string representing a line of text from refmappingshopgroup.txt in the Media.pk2 into a MappingShopGroup object
//
// [param] line   A line of text
// [return]       A populated MappingShopGroup object
pk2::ref::MappingShopGroup parseMappingShopGroupLine(const std::string &line);

// Parses string representing a line of text from refmappingshopwithtab.txt in the Media.pk2 into a MappingShopWithTab object
//
// [param] line   A line of text
// [return]       A populated MappingShopWithTab object
pk2::ref::MappingShopWithTab parseMappingShopWithTabLine(const std::string &line);

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