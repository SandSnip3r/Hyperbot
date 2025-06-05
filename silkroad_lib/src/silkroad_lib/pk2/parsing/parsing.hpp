#ifndef PK2_PARSING_HPP_
#define PK2_PARSING_HPP_

#include <silkroad_lib/pk2/divisionInfo.hpp>
#include <silkroad_lib/pk2/ref/character.hpp>
#include <silkroad_lib/pk2/ref/item.hpp>
#include <silkroad_lib/pk2/ref/level.hpp>
#include <silkroad_lib/pk2/ref/magicOption.hpp>
#include <silkroad_lib/pk2/ref/mappingShopGroup.hpp>
#include <silkroad_lib/pk2/ref/mappingShopWithTab.hpp>
#include <silkroad_lib/pk2/ref/mastery.hpp>
#include <silkroad_lib/pk2/ref/region.hpp>
#include <silkroad_lib/pk2/ref/scrapOfPackageItem.hpp>
#include <silkroad_lib/pk2/ref/shopGood.hpp>
#include <silkroad_lib/pk2/ref/shopGroup.hpp>
#include <silkroad_lib/pk2/ref/shopTab.hpp>
#include <silkroad_lib/pk2/ref/skill.hpp>
#include <silkroad_lib/pk2/ref/teleport.hpp>
#include <silkroad_lib/pk2/ref/text.hpp>
#include <silkroad_lib/pk2/ref/textZoneName.hpp>

#include <charconv>
#include <filesystem>
#include <ostream>
#include <string_view>
#include <string>
#include <vector>

namespace sro::pk2::parsing {

// Transforms raw pk2 data into a std::string
//
// [param] data   Data from a PK2Entry
// [return]       data as an std::string
std::string fileDataToString(const std::vector<uint8_t> &data);

// Decrypts skilldata_<>enc.txt files
//
// [param] data   File data to decrypt
void decryptSkillData(std::vector<uint8_t> &data);

// Validates if the line of magic option data is valid
//
// [param] line   Line from PK2 file representing magic option data
// [return]       Whether the line is valid or not
bool isValidMagicOptionDataLine(const std::string &line);

// Validates if the line of level data is valid
//
// [param] line   Line from PK2 file representing level data
// [return]       Whether the line is valid or not
bool isValidLevelDataLine(const std::string &line);

// Validates if the line of refRegion is valid
//
// [param] line   Line from PK2 file representing a ref region
// [return]       Whether the line is valid or not
bool isValidRefRegionLine(const std::string &line);

// Validates if the line of teleport building data is valid
//
// [param] line   Line from PK2 file representing a teleport building
// [return]       Whether the line is valid or not
bool isValidTeleportbuildingLine(const std::string &line);

// Validates if the line of ScrapOfPackageItem data is valid
//
// [param] line   Line from PK2 file representing a ScrapOfPackageItem
// [return]       Whether the line is valid or not
bool isValidScrapOfPackageItemLine(const std::string &line);

// Validates if the line of ShopTab data is valid
//
// [param] line   Line from PK2 file representing a ShopTab
// [return]       Whether the line is valid or not
bool isValidShopTabLine(const std::string &line);

// Validates if the line of ShopGroup data is valid
//
// [param] line   Line from PK2 file representing a ShopGroup
// [return]       Whether the line is valid or not
bool isValidShopGroupLine(const std::string &line);

// Validates if the line of ShopGood data is valid
//
// [param] line   Line from PK2 file representing a ShopGood
// [return]       Whether the line is valid or not
bool isValidShopGoodLine(const std::string &line);

// Validates if the line of MappingShopGroup data is valid
//
// [param] line   Line from PK2 file representing a MappingShopGroup
// [return]       Whether the line is valid or not
bool isValidMappingShopGroupLine(const std::string &line);

// Validates if the line of MappingShopWithTab data is valid
//
// [param] line   Line from PK2 file representing a MappingShopWithTab
// [return]       Whether the line is valid or not
bool isValidMappingShopWithTabLine(const std::string &line);

// Parses string representing a line of text from skilldata_xxxx.txt in the Media.pk2 into a Skill object
ref::Skill parseSkilldataLine(const std::vector<std::string_view> &linePieces);

// Parses string representing a line of text from characterdata_xxxx.txt in the Media.pk2 into an Character object
ref::Character parseCharacterdataLine(const std::vector<std::string_view> &linePieces);

// Parses string representing a line of text from itemdata_xxxx.txt in the Media.pk2 into an Item object
ref::Item parseItemdataLine(const std::vector<std::string_view> &linePieces);

// Parses string representing a line of text from magicoption.txt in the Media.pk2 into a MagicOption object
//
// [param] line   A line of text
// [return]       A populated MagicOption object
ref::MagicOption parseMagicOptionDataLine(const std::string &line);

// Parses string representing a line of text from leveldata.txt in the Media.pk2 into a Level object
//
// [param] line   A line of text
// [return]       A populated Level object
ref::Level parseLevelDataLine(const std::string &line);

// Parses string representing a line of text from refregion.txt in the Media.pk2 into a Region object
//
// [param] line   A line of text
// [return]       A populated Region object
ref::Region parseRefRegionLine(const std::string &line);

// Parses string representing a line of text from teleportbuilding.txt in the Media.pk2 into a Teleport object
//
// [param] line   A line of text
// [return]       A populated Teleport object
ref::Teleport parseTeleportbuildingLine(const std::string &line);

// Parses string representing a line of text from refscrapofpackageitem.txt in the Media.pk2 into a ScrapOfPackageItem object
//
// [param] line   A line of text
// [return]       A populated ScrapOfPackageItem object
ref::ScrapOfPackageItem parseScrapOfPackageItemLine(const std::string &line);

// Parses string representing a line of text from refshoptab.txt in the Media.pk2 into a ShopTab object
//
// [param] line   A line of text
// [return]       A populated ShopTab object
ref::ShopTab parseShopTabLine(const std::string &line);

// Parses string representing a line of text from refshopgroup.txt in the Media.pk2 into a ShopGroup object
//
// [param] line   A line of text
// [return]       A populated ShopGroup object
ref::ShopGroup parseShopGroupLine(const std::string &line);

// Parses string representing a line of text from refshopgoods.txt in the Media.pk2 into a ShopGood object
//
// [param] line   A line of text
// [return]       A populated ShopGood object
ref::ShopGood parseShopGoodLine(const std::string &line);

// Parses string representing a line of text from refmappingshopgroup.txt in the Media.pk2 into a MappingShopGroup object
//
// [param] line   A line of text
// [return]       A populated MappingShopGroup object
ref::MappingShopGroup parseMappingShopGroupLine(const std::string &line);

// Parses string representing a line of text from refmappingshopwithtab.txt in the Media.pk2 into a MappingShopWithTab object
//
// [param] line   A line of text
// [return]       A populated MappingShopWithTab object
ref::MappingShopWithTab parseMappingShopWithTabLine(const std::string &line);

// Parses string representing a line of text from textzonename.txt in the Media.pk2 into a TextZoneName object
ref::TextZoneName parseTextZoneNameLine(const std::vector<std::string_view> &linePieces);

// Parses string representing a line of text from textdata_equip&skill.txt or textdata_object.txt in the Media.pk2 into a Text object
//  Note: This isn't the only type of data in these files
ref::Text parseTextLine(const std::vector<std::string_view> &linePieces);

// Parses string representing a line of text from skillmasterydata.txt in the Media.pk2 into a Mastery object
//  Note: This isn't the only type of data in these files
ref::Mastery parseMasteryLine(const std::vector<std::string_view> &linePieces);

// Parses raw pk2 data into a gateway server port
//
// [param] data   Data from the GATEWAYPORT.TXT PK2Entry
// [return]       uint16_t gateway server port
uint16_t parseGatePort(const std::vector<uint8_t> &data);

// Parses raw pk2 data into DivisionInfo object
//
// [param] data   Data from the DIVISIONINFO.TXT PK2Entry
// [return]       Populated DivisionInfo object
DivisionInfo parseDivisionInfo(const std::vector<uint8_t> &data);

// Splits a string into pieces which do not contain the delimiter
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
void parse(std::string_view data, T &result) {
  if constexpr (std::is_integral_v<T> || std::is_enum_v<T> || std::is_floating_point_v<T>) {
    const char* begin = data.data();
    const char* end   = data.data() + data.size();
    auto [ptr, ec] = std::from_chars(begin, end, result);
    if (ec != std::errc{}) {
      // How many characters did std::from_chars successfully consume?
      std::size_t offset = ptr - begin;

      // Convert std::errc into a human‐readable string:
      std::string errMsg = std::make_error_code(ec).message();

      // Include the type name of T (mangled, but still useful for debug):
      const char* typeName = typeid(T).name();

      throw std::runtime_error(
        "parse<" + std::string(typeName) + ">: failed to parse integer from \"" +
        std::string(data) + "\"\n"
        "  error: " + errMsg + "\n"
        "  characters consumed: " + std::to_string(offset) + " of " +
        std::to_string(data.size()) + "\n"
        "  remaining input (from failure point): \"" +
        std::string(ptr, end) + "\""
      );
    }
  } else if constexpr (std::is_same_v<T, std::string>) {
    result = std::string(data);
  } else {
    static_assert(false, "Unsupported type for StringViewStreamParser::read");
  }
}

template<typename T>
const char* parse(const char *begin, T &result);
template<>
const char* parse<std::string>(const char *begin, std::string &result);
template<>
const char* parse<uint8_t>(const char *begin, uint8_t &result);
template<>
const char* parse<int16_t>(const char *begin, int16_t &result);
template<>
const char* parse<int32_t>(const char *begin, int32_t &result);
template<>
const char* parse<int64_t>(const char *begin, int64_t &result);
template<>
const char* parse<float>(const char *begin, float &result);

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

} // namespace sro::pk2::parsing

#endif // PK2_PARSING_HPP_