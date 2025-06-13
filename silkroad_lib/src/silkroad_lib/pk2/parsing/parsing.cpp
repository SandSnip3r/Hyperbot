// #define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING true

#include <silkroad_lib/pk2/parsing/parsing.hpp>

#include <absl/log/log.h>

#include <array>
#include <codecvt>
#include <cstring>
#include <sstream>

namespace sro::pk2::parsing {

namespace {

/**
 * Append a single Unicode codepoint `cp` to `out` in UTF-8 form.
 * Assumes cp <= 0x10FFFF (valid Unicode range).
 */
void appendCodepoint(std::string& out, uint32_t cp) {
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

/**
 * Convert a sequence of little-endian UTF-16 code units to a UTF-8 string.
 * Handles surrogate pairs but does not attempt any advanced Unicode
 * normalization, error detection, etc.
 */
std::string utf16leToUtf8(const uint16_t* data, size_t count) {
  std::string out;
  out.reserve(count); // Minimum safe reservation

  for (size_t i = 0; i < count; ++i) {
    uint16_t w1 = data[i];

    // Check for a leading surrogate
    if (w1 >= 0xD800 && w1 <= 0xDBFF) {
      // Make sure we have a trailing surrogate
      if (i + 1 < count) {
        uint16_t w2 = data[i + 1];
        if (w2 >= 0xDC00 && w2 <= 0xDFFF) {
          // Valid surrogate pair
          uint32_t codePoint = 0x10000
                              + ((static_cast<uint32_t>(w1) & 0x3FF) << 10)
                              + (static_cast<uint32_t>(w2) & 0x3FF);
          appendCodepoint(out, codePoint);
          ++i; // Skip the next code unit (w2) as well
          continue;
        }
      }
      // If we get here: malformed surrogate pair; handle or throw
      // For simplicity, we’ll just treat it as a replacement character U+FFFD
      appendCodepoint(out, 0xFFFD);
    } else if (w1 >= 0xDC00 && w1 <= 0xDFFF) {
      // Trailing surrogate without a leading; also malformed
      appendCodepoint(out, 0xFFFD);
    } else {
      // Basic Multilingual Plane code unit
      appendCodepoint(out, w1);
    }
  }
  return out;
}

} // anonymous namespace

std::string fileDataToString(const std::vector<uint8_t> &data) {
  // codecvt is removed in C++20.
  // If using C++20, maybe use https://github.com/nemtrif/utfcpp instead.
  if (data.size() % 2 != 0) {
    throw std::runtime_error("File size is not a multiple of 2 bytes.");
  }

  size_t start = 0;
  // Check for BOM 0xFF 0xFE
  if (data.size() >= 2 && static_cast<uint8_t>(data[0]) == 0xFF && static_cast<uint8_t>(data[1]) == 0xFE) {
    start = 2;
  }

  std::u16string result;
  for (size_t i = start; i < data.size(); i += 2) {
    char16_t ch = static_cast<unsigned char>(data[i]) |
                  (static_cast<unsigned char>(data[i + 1]) << 8);
    result.push_back(ch);
  }

  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> conv;
  return conv.to_bytes(result);
}

namespace {
class Decryptor {
public:
  static constexpr int32_t baseKey_ = 0x8C1F;
  int16_t getRollingKey() {
    const int16_t result = kHashTable1_[currentKey_ % 0xA7] - kHashTable2_[currentKey_ % 0x1EF];
    ++currentKey_;
    return result;
  }
private:
  int32_t currentKey_{baseKey_};
  static const std::array<uint8_t,256> kHashTable1_;
  static const std::array<uint8_t,496> kHashTable2_;
};
const std::array<uint8_t,256> Decryptor::kHashTable1_ = {
  0x07, 0x83, 0xBC, 0xEE, 0x4B, 0x79, 0x19, 0xB6, 0x2A, 0x53, 0x4F, 0x3A, 0xCF, 0x71, 0xE5, 0x3C,
  0x2D, 0x18, 0x14, 0xCB, 0xB6, 0xBC, 0xAA, 0x9A, 0x31, 0x42, 0x3A, 0x13, 0x42, 0xC9, 0x63, 0xFC,
  0x54, 0x1D, 0xF2, 0xC1, 0x8A, 0xDD, 0x1C, 0xB3, 0x52, 0xEA, 0x9B, 0xD7, 0xC4, 0xBA, 0xF8, 0x12,
  0x74, 0x92, 0x30, 0xC9, 0xD6, 0x56, 0x15, 0x52, 0x53, 0x60, 0x11, 0x33, 0xC5, 0x9D, 0x30, 0x9A,
  0xE5, 0xD2, 0x93, 0x99, 0xEB, 0xCF, 0xAA, 0x79, 0xE3, 0x78, 0x6A, 0xB9, 0x02, 0xE0, 0xCE, 0x8E,
  0xF3, 0x63, 0x5A, 0x73, 0x74, 0xF3, 0x72, 0xAA, 0x2C, 0x9F, 0xBB, 0x33, 0x91, 0xDE, 0x5F, 0x91,
  0x66, 0x48, 0xD1, 0x7A, 0xFD, 0x3F, 0x91, 0x3E, 0x5D, 0x22, 0xEC, 0xEF, 0x7C, 0xA5, 0x43, 0xC0,
  0x1D, 0x4F, 0x60, 0x7F, 0x0B, 0x4A, 0x4B, 0x2A, 0x43, 0x06, 0x46, 0x14, 0x45, 0xD0, 0xC5, 0x83,
  0x92, 0xE4, 0x16, 0xD0, 0xA3, 0xA1, 0x13, 0xDA, 0xD1, 0x51, 0x07, 0xEB, 0x7D, 0xCE, 0xA5, 0xDB,
  0x78, 0xE0, 0xC1, 0x0B, 0xE5, 0x8E, 0x1C, 0x7C, 0xB4, 0xDF, 0xED, 0xB8, 0x53, 0xBA, 0x2C, 0xB5,
  0xBB, 0x56, 0xFB, 0x68, 0x95, 0x6E, 0x65, 0x00, 0x60, 0xBA, 0xE3, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x9C, 0xB5, 0xD5, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2E, 0x3F, 0x41, 0x56,
  0x43, 0x45, 0x53, 0x63, 0x72, 0x69, 0x70, 0x74, 0x40, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x64, 0xBB, 0xE3, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
const std::array<uint8_t,496> Decryptor::kHashTable2_ = {
  0x0D, 0x05, 0x90, 0x41, 0xF9, 0xD0, 0x65, 0xBF, 0xF9, 0x0B, 0x15, 0x93, 0x80, 0xFB, 0x01, 0x02,
  0xB6, 0x08, 0xC4, 0x3C, 0xC1, 0x49, 0x94, 0x4D, 0xCE, 0x1D, 0xFD, 0x69, 0xEA, 0x19, 0xC9, 0x57,
  0x9C, 0x4D, 0x84, 0x62, 0xE3, 0x67, 0xF9, 0x87, 0xF4, 0xF9, 0x93, 0xDA, 0xE5, 0x15, 0xF1, 0x4C,
  0xA4, 0xEC, 0xBC, 0xCF, 0xDD, 0xB3, 0x6F, 0x04, 0x3D, 0x70, 0x1C, 0x74, 0x21, 0x6B, 0x00, 0x71,
  0x31, 0x7F, 0x54, 0xB3, 0x72, 0x6C, 0xAA, 0x42, 0xC1, 0x78, 0x61, 0x3E, 0xD5, 0xF2, 0xE1, 0x27,
  0x36, 0x71, 0x3A, 0x25, 0x36, 0x57, 0xD1, 0xF8, 0x70, 0x86, 0xBD, 0x0E, 0x58, 0xB3, 0x76, 0x6D,
  0xC3, 0x50, 0xF6, 0x6C, 0xA0, 0x10, 0x06, 0x64, 0xA2, 0xD6, 0x2C, 0xD4, 0x27, 0x30, 0xA5, 0x36,
  0x1C, 0x1E, 0x3E, 0x58, 0x9D, 0x59, 0x76, 0x9D, 0xA7, 0x42, 0x5A, 0xF0, 0x00, 0xBC, 0x69, 0x31,
  0x40, 0x1E, 0xFA, 0x09, 0x1D, 0xE7, 0xEE, 0xE4, 0x54, 0x89, 0x36, 0x7C, 0x67, 0xC8, 0x65, 0x22,
  0x7E, 0xA3, 0x60, 0x44, 0x1E, 0xBC, 0x68, 0x6F, 0x15, 0x2A, 0xFD, 0x9D, 0x3F, 0x36, 0x6B, 0x28,
  0x06, 0x67, 0xFE, 0xC6, 0x49, 0x6B, 0x9B, 0x3F, 0x80, 0x2A, 0xD2, 0xD4, 0xD3, 0x20, 0x1B, 0x96,
  0xF4, 0xD2, 0xCA, 0x8C, 0x74, 0xEE, 0x0B, 0x6A, 0xE1, 0xE9, 0xC6, 0xD2, 0x6E, 0x33, 0x63, 0xC0,
  0xE9, 0xD0, 0x37, 0xA9, 0x3C, 0xF7, 0x18, 0xF2, 0x4A, 0x74, 0xEC, 0x41, 0x61, 0x7A, 0x19, 0x47,
  0x8F, 0xA0, 0xBB, 0x94, 0x8F, 0x3D, 0x11, 0x11, 0x26, 0xCF, 0x69, 0x18, 0x1B, 0x2C, 0x87, 0x6D,
  0xB3, 0x22, 0x6C, 0x78, 0x41, 0xCC, 0xC2, 0x84, 0xC5, 0xCB, 0x01, 0x6A, 0x37, 0x00, 0x01, 0x65,
  0x4F, 0xA7, 0x85, 0x85, 0x15, 0x59, 0x05, 0x67, 0xF2, 0x4F, 0xAB, 0xB7, 0x88, 0xFA, 0x69, 0x24,
  0x9E, 0xC6, 0x7B, 0x3F, 0xD5, 0x0E, 0x4D, 0x7B, 0xFB, 0xB1, 0x21, 0x3C, 0xB0, 0xC0, 0xCB, 0x2C,
  0xAA, 0x26, 0x8D, 0xCC, 0xDD, 0xDA, 0xC1, 0xF8, 0xCA, 0x7F, 0x6A, 0x3F, 0x2A, 0x61, 0xE7, 0x60,
  0x5C, 0xCE, 0xD3, 0x4C, 0xAC, 0x45, 0x40, 0x62, 0xEA, 0x51, 0xF1, 0x66, 0x5D, 0x2C, 0x45, 0xD6,
  0x8B, 0x7D, 0xCE, 0x9C, 0xF5, 0xBB, 0xF7, 0x52, 0x24, 0x1A, 0x13, 0x02, 0x2B, 0x00, 0xBB, 0xA1,
  0x8F, 0x6E, 0x7A, 0x33, 0xAD, 0x5F, 0xF4, 0x4A, 0x82, 0x76, 0xAB, 0xDE, 0x80, 0x98, 0x8B, 0x26,
  0x4F, 0x33, 0xD8, 0x68, 0x1E, 0xD9, 0xAE, 0x06, 0x6B, 0x7E, 0xA9, 0x95, 0x67, 0x60, 0xEB, 0xE8,
  0xD0, 0x7D, 0x07, 0x4B, 0xF1, 0xAA, 0x9A, 0xC5, 0x29, 0x93, 0x9D, 0x5C, 0x92, 0x3F, 0x15, 0xDE,
  0x48, 0xF1, 0xCA, 0xEA, 0xC9, 0x78, 0x3C, 0x28, 0x7E, 0xB0, 0x46, 0xD3, 0x71, 0x6C, 0xD7, 0xBD,
  0x2C, 0xF7, 0x25, 0x2F, 0xC7, 0xDD, 0xB4, 0x6D, 0x35, 0xBB, 0xA7, 0xDA, 0x3E, 0x3D, 0xA7, 0xCA,
  0xBD, 0x87, 0xDD, 0x9F, 0x22, 0x3D, 0x50, 0xD2, 0x30, 0xD5, 0x14, 0x5B, 0x8F, 0xF4, 0xAF, 0xAA,
  0xA0, 0xFC, 0x17, 0x3D, 0x33, 0x10, 0x99, 0xDC, 0x76, 0xA9, 0x40, 0x1B, 0x64, 0x14, 0xDF, 0x35,
  0x68, 0x66, 0x5B, 0x49, 0x05, 0x33, 0x68, 0x26, 0xC8, 0xBA, 0xD1, 0x8D, 0x39, 0x2B, 0xFB, 0x3E,
  0x24, 0x52, 0x2F, 0x9A, 0x69, 0xBC, 0xF2, 0xB2, 0xAC, 0xB8, 0xEF, 0xA1, 0x17, 0x29, 0x2D, 0xEE,
  0xF5, 0x23, 0x21, 0xEC, 0x81, 0xC7, 0x5B, 0xC0, 0x82, 0xCC, 0xD2, 0x91, 0x9D, 0x29, 0x93, 0x0C,
  0x9D, 0x5D, 0x57, 0xAD, 0xD4, 0xC6, 0x40, 0x93, 0x8D, 0xE9, 0xD3, 0x35, 0x9D, 0xC6, 0xD3, 0x00,
};
} // namespace (anonymous)

void decryptSkillData(std::vector<uint8_t> &data) {
  // The encrypted data files seem to trail with 11 characters of metadata, we will throw this away
  //  Remember, characters are 2 bytes each because of UTF-16
  // TODO: Figure out what this is
  constexpr int kTrailingMetaDataSize = 22;
  data.resize(data.size()-kTrailingMetaDataSize);
  Decryptor decryptor;
  for (int i=0; i<data.size(); ++i) {
    data[i] = static_cast<char>(data[i]) + decryptor.getRollingKey();
  }
}

namespace {
bool isValidLine(const int fieldCount, const std::string &line) {
  // TODO: Evaluate more robust way to validate
  if (line.size() < fieldCount*2+1) {
    return false;
  }
  if (line[0] == '/' && line[1] == '/') {
    return false;
  }
  // Check for at least fieldCount-1 tabs, since we know files have fields separated by tabs
  int tabCount=0;
  for (int i=0; i<line.size(); ++i) {
    if (line[i] == '\t') {
      ++tabCount;
    }
    if (tabCount == fieldCount-1) {
      // Have enough tabs
      return true;
    }
  }
  // Didnt find enough tabs
  return false;
}
} // namespace (anonymous)

bool isValidCharacterdataLine(const std::string &line) {
  constexpr int kDataCount = 104;
  return isValidLine(kDataCount, line);
}

bool isValidItemdataLine(const std::string &line) {
  constexpr int kDataCount = 160;
  return isValidLine(kDataCount, line);
}

bool isValidMagicOptionDataLine(const std::string &line) {
  constexpr int kDataCount = 49;
  return isValidLine(kDataCount, line);
}

bool isValidLevelDataLine(const std::string &line) {
  constexpr int kDataCount = 9;
  return isValidLine(kDataCount, line);
}

bool isValidRefRegionLine(const std::string &line) {
  constexpr int kDataCount = 21;
  return isValidLine(kDataCount, line);
}

bool isValidSkilldataLine(const std::string &line) {
  constexpr int kDataCount = 117;
  return isValidLine(kDataCount, line);
}

bool isValidTeleportbuildingLine(const std::string &line) {
  constexpr int kDataCount = 58;
  return isValidLine(kDataCount, line);
}

bool isValidScrapOfPackageItemLine(const std::string &line) {
  constexpr int kDataCount = 29;
  return isValidLine(kDataCount, line);
}

bool isValidShopTabLine(const std::string &line) {
  constexpr int kDataCount = 6;
  return isValidLine(kDataCount, line);
}

bool isValidShopGroupLine(const std::string &line) {
  constexpr int kDataCount = 13;
  return isValidLine(kDataCount, line);
}

bool isValidShopGoodLine(const std::string &line) {
  constexpr int kDataCount = 13;
  return isValidLine(kDataCount, line);
}
bool isValidMappingShopGroupLine(const std::string &line) {
  constexpr int kDataCount = 4;
  return isValidLine(kDataCount, line);
}

bool isValidMappingShopWithTabLine(const std::string &line) {
  constexpr int kDataCount = 4;
  return isValidLine(kDataCount, line);
}

bool isValidTextDataLine(const std::string &line) {
  constexpr int kDataCount = 15;
  return isValidLine(kDataCount, line);
}

bool isValidMasteryLine(const std::string &line) {
  constexpr int kDataCount = 13;
  return isValidLine(kDataCount, line);
}

ref::Character parseCharacterdataLine(const std::vector<absl::string_view> &linePieces) {
  ref::Character character;
  parse(linePieces[0], character.service);
  parse(linePieces[1], character.id);
  parse(linePieces[2], character.codeName128);
  parse(linePieces[3], character.objName128);
  parse(linePieces[4], character.orgObjCodeName128);
  parse(linePieces[5], character.nameStrID128);
  parse(linePieces[6], character.descStrID128);
  parse(linePieces[7], character.cashItem);
  parse(linePieces[8], character.bionic);
  parse(linePieces[9], character.typeId1);
  parse(linePieces[10], character.typeId2);
  parse(linePieces[11], character.typeId3);
  parse(linePieces[12], character.typeId4);
  parse(linePieces[13], character.decayTime);
  parse(linePieces[14], character.country);
  parse(linePieces[15], character.rarity);
  parse(linePieces[16], character.canTrade);
  parse(linePieces[17], character.canSell);
  parse(linePieces[18], character.canBuy);
  parse(linePieces[19], character.canBorrow);
  parse(linePieces[20], character.canDrop);
  parse(linePieces[21], character.canPick);
  parse(linePieces[22], character.canRepair);
  parse(linePieces[23], character.canRevive);
  parse(linePieces[24], character.canUse);
  parse(linePieces[25], character.canThrow);
  parse(linePieces[26], character.price);
  parse(linePieces[27], character.costRepair);
  parse(linePieces[28], character.costRevive);
  parse(linePieces[29], character.costBorrow);
  parse(linePieces[30], character.keepingFee);
  parse(linePieces[31], character.sellPrice);
  parse(linePieces[32], character.reqLevelType1);
  parse(linePieces[33], character.reqLevel1);
  parse(linePieces[34], character.reqLevelType2);
  parse(linePieces[35], character.reqLevel2);
  parse(linePieces[36], character.reqLevelType3);
  parse(linePieces[37], character.reqLevel3);
  parse(linePieces[38], character.reqLevelType4);
  parse(linePieces[39], character.reqLevel4);
  parse(linePieces[40], character.maxContain);
  parse(linePieces[41], character.regionID);
  parse(linePieces[42], character.dir);
  parse(linePieces[43], character.offsetX);
  parse(linePieces[44], character.offsetY);
  parse(linePieces[45], character.offsetZ);
  parse(linePieces[46], character.speed1);
  parse(linePieces[47], character.speed2);
  parse(linePieces[48], character.scale);
  parse(linePieces[49], character.bCHeight);
  parse(linePieces[50], character.bCRadius);
  parse(linePieces[51], character.eventID);
  parse(linePieces[52], character.assocFileObj128);
  parse(linePieces[53], character.assocFileDrop128);
  parse(linePieces[54], character.assocFileIcon128);
  parse(linePieces[55], character.assocFile1_128);
  parse(linePieces[56], character.assocFile2_128);
  parse(linePieces[57], character.lvl);
  parse(linePieces[58], character.charGender);
  parse(linePieces[59], character.maxHp);
  parse(linePieces[60], character.maxMp);
  parse(linePieces[61], character.inventorySize);
  parse(linePieces[62], character.canStore_TID1);
  parse(linePieces[63], character.canStore_TID2);
  parse(linePieces[64], character.canStore_TID3);
  parse(linePieces[65], character.canStore_TID4);
  parse(linePieces[66], character.canBeVehicle);
  parse(linePieces[67], character.canControl);
  parse(linePieces[68], character.damagePortion);
  parse(linePieces[69], character.maxPassenger);
  parse(linePieces[70], character.assocTactics);
  parse(linePieces[71], character.pd);
  parse(linePieces[72], character.md);
  parse(linePieces[73], character.par);
  parse(linePieces[74], character.mar);
  parse(linePieces[75], character.er);
  parse(linePieces[76], character.br);
  parse(linePieces[77], character.hr);
  parse(linePieces[78], character.chr);
  parse(linePieces[79], character.expToGive);
  parse(linePieces[80], character.creepType);
  parse(linePieces[81], character.knockdown);
  parse(linePieces[82], character.kO_RecoverTime);
  parse(linePieces[83], character.defaultSkill_1);
  parse(linePieces[84], character.defaultSkill_2);
  parse(linePieces[85], character.defaultSkill_3);
  parse(linePieces[86], character.defaultSkill_4);
  parse(linePieces[87], character.defaultSkill_5);
  parse(linePieces[88], character.defaultSkill_6);
  parse(linePieces[89], character.defaultSkill_7);
  parse(linePieces[90], character.defaultSkill_8);
  parse(linePieces[91], character.defaultSkill_9);
  parse(linePieces[92], character.defaultSkill_10);
  parse(linePieces[93], character.textureType);
  parse(linePieces[94], character.except_1);
  parse(linePieces[95], character.except_2);
  parse(linePieces[96], character.except_3);
  parse(linePieces[97], character.except_4);
  parse(linePieces[98], character.except_5);
  parse(linePieces[99], character.except_6);
  parse(linePieces[100], character.except_7);
  parse(linePieces[101], character.except_8);
  parse(linePieces[102], character.except_9);
  parse(linePieces[103], character.except_10);
  return character;
}

ref::Item parseItemdataLine(const std::vector<absl::string_view> &linePieces) {
  ref::Item item;
  parse(linePieces[0], item.service);
  parse(linePieces[1], item.id);
  parse(linePieces[2], item.codeName128);
  parse(linePieces[3], item.objName128);
  parse(linePieces[4], item.orgObjCodeName128);
  parse(linePieces[5], item.nameStrID128);
  parse(linePieces[6], item.descStrID128);
  parse(linePieces[7], item.cashItem);
  parse(linePieces[8], item.bionic);
  parse(linePieces[9], item.typeId1);
  parse(linePieces[10], item.typeId2);
  parse(linePieces[11], item.typeId3);
  parse(linePieces[12], item.typeId4);
  parse(linePieces[13], item.decayTime);
  parse(linePieces[14], item.country);
  parse(linePieces[15], item.rarity);
  parse(linePieces[16], item.canTrade);
  parse(linePieces[17], item.canSell);
  parse(linePieces[18], item.canBuy);
  parse(linePieces[19], item.canBorrow);
  parse(linePieces[20], item.canDrop);
  parse(linePieces[21], item.canPick);
  parse(linePieces[22], item.canRepair);
  parse(linePieces[23], item.canRevive);
  parse(linePieces[24], item.canUse);
  parse(linePieces[25], item.canThrow);
  parse(linePieces[26], item.price);
  parse(linePieces[27], item.costRepair);
  parse(linePieces[28], item.costRevive);
  parse(linePieces[29], item.costBorrow);
  parse(linePieces[30], item.keepingFee);
  parse(linePieces[31], item.sellPrice);
  parse(linePieces[32], item.reqLevelType1);
  parse(linePieces[33], item.reqLevel1);
  parse(linePieces[34], item.reqLevelType2);
  parse(linePieces[35], item.reqLevel2);
  parse(linePieces[36], item.reqLevelType3);
  parse(linePieces[37], item.reqLevel3);
  parse(linePieces[38], item.reqLevelType4);
  parse(linePieces[39], item.reqLevel4);
  parse(linePieces[40], item.maxContain);
  parse(linePieces[41], item.regionID);
  parse(linePieces[42], item.dir);
  parse(linePieces[43], item.offsetX);
  parse(linePieces[44], item.offsetY);
  parse(linePieces[45], item.offsetZ);
  parse(linePieces[46], item.speed1);
  parse(linePieces[47], item.speed2);
  parse(linePieces[48], item.scale);
  parse(linePieces[49], item.bCHeight);
  parse(linePieces[50], item.bCRadius);
  parse(linePieces[51], item.eventID);
  parse(linePieces[52], item.assocFileObj128);
  parse(linePieces[53], item.assocFileDrop128);
  parse(linePieces[54], item.assocFileIcon128);
  parse(linePieces[55], item.assocFile1_128);
  parse(linePieces[56], item.assocFile2_128);
  parse(linePieces[57], item.maxStack);
  parse(linePieces[58], item.reqGender);
  parse(linePieces[59], item.reqStr);
  parse(linePieces[60], item.reqInt);
  parse(linePieces[61], item.itemClass);
  parse(linePieces[62], item.setID);
  parse(linePieces[63], item.dur_L);
  parse(linePieces[64], item.dur_U);
  parse(linePieces[65], item.pd_L);
  parse(linePieces[66], item.pd_U);
  parse(linePieces[67], item.pdInc);
  parse(linePieces[68], item.er_L);
  parse(linePieces[69], item.er_U);
  parse(linePieces[70], item.eRInc);
  parse(linePieces[71], item.par_L);
  parse(linePieces[72], item.par_U);
  parse(linePieces[73], item.parInc);
  parse(linePieces[74], item.br_L);
  parse(linePieces[75], item.br_U);
  parse(linePieces[76], item.md_L);
  parse(linePieces[77], item.md_U);
  parse(linePieces[78], item.mdInc);
  parse(linePieces[79], item.mar_L);
  parse(linePieces[80], item.mar_U);
  parse(linePieces[81], item.marInc);
  parse(linePieces[82], item.pdStr_L);
  parse(linePieces[83], item.pdStr_U);
  parse(linePieces[84], item.mdInt_L);
  parse(linePieces[85], item.mdInt_U);
  parse(linePieces[86], item.quivered);
  parse(linePieces[87], item.ammo1_TID4);
  parse(linePieces[88], item.ammo2_TID4);
  parse(linePieces[89], item.ammo3_TID4);
  parse(linePieces[90], item.ammo4_TID4);
  parse(linePieces[91], item.ammo5_TID4);
  parse(linePieces[92], item.speedClass);
  parse(linePieces[93], item.twoHanded);
  parse(linePieces[94], item.range);
  parse(linePieces[95], item.pAttackMin_L);
  parse(linePieces[96], item.pAttackMin_U);
  parse(linePieces[97], item.pAttackMax_L);
  parse(linePieces[98], item.pAttackMax_U);
  parse(linePieces[99], item.pAttackInc);
  parse(linePieces[100], item.mAttackMin_L);
  parse(linePieces[101], item.mAttackMin_U);
  parse(linePieces[102], item.mAttackMax_L);
  parse(linePieces[103], item.mAttackMax_U);
  parse(linePieces[104], item.mAttackInc);
  parse(linePieces[105], item.paStrMin_L);
  parse(linePieces[106], item.paStrMin_U);
  parse(linePieces[107], item.paStrMax_L);
  parse(linePieces[108], item.paStrMax_U);
  parse(linePieces[109], item.maInt_Min_L);
  parse(linePieces[110], item.maInt_Min_U);
  parse(linePieces[111], item.maInt_Max_L);
  parse(linePieces[112], item.maInt_Max_U);
  parse(linePieces[113], item.hr_L);
  parse(linePieces[114], item.hr_U);
  parse(linePieces[115], item.hRInc);
  parse(linePieces[116], item.cHR_L);
  parse(linePieces[117], item.cHR_U);
  parse(linePieces[118], item.param1);
  parse(linePieces[119], item.desc1_128);
  parse(linePieces[120], item.param2);
  parse(linePieces[121], item.desc2_128);
  parse(linePieces[122], item.param3);
  parse(linePieces[123], item.desc3_128);
  parse(linePieces[124], item.param4);
  parse(linePieces[125], item.desc4_128);
  parse(linePieces[126], item.param5);
  parse(linePieces[127], item.desc5_128);
  parse(linePieces[128], item.param6);
  parse(linePieces[129], item.desc6_128);
  parse(linePieces[130], item.param7);
  parse(linePieces[131], item.desc7_128);
  parse(linePieces[132], item.param8);
  parse(linePieces[133], item.desc8_128);
  parse(linePieces[134], item.param9);
  parse(linePieces[135], item.desc9_128);
  parse(linePieces[136], item.param10);
  parse(linePieces[137], item.desc10_128);
  parse(linePieces[138], item.param11);
  parse(linePieces[139], item.desc11_128);
  parse(linePieces[140], item.param12);
  parse(linePieces[141], item.desc12_128);
  parse(linePieces[142], item.param13);
  parse(linePieces[143], item.desc13_128);
  parse(linePieces[144], item.param14);
  parse(linePieces[145], item.desc14_128);
  parse(linePieces[146], item.param15);
  parse(linePieces[147], item.desc15_128);
  parse(linePieces[148], item.param16);
  parse(linePieces[149], item.desc16_128);
  parse(linePieces[150], item.param17);
  parse(linePieces[151], item.desc17_128);
  parse(linePieces[152], item.param18);
  parse(linePieces[153], item.desc18_128);
  parse(linePieces[154], item.param19);
  parse(linePieces[155], item.desc19_128);
  parse(linePieces[156], item.param20);
  parse(linePieces[157], item.desc20_128);
  parse(linePieces[158], item.maxMagicOptCount);
  parse(linePieces[159], item.childItemCount);
  return item;
}

ref::MagicOption parseMagicOptionDataLine(const std::string &line) {
  ref::MagicOption magOpt;
  const char *ptr = line.data();
  ptr = parse(ptr, magOpt.service);
  ptr = parse(ptr, magOpt.id);
  ptr = parse(ptr, magOpt.mOptName128);
  ptr = parse(ptr, magOpt.attrType);
  ptr = parse(ptr, magOpt.mLevel);
  ptr = parse(ptr, magOpt.prob);
  ptr = parse(ptr, magOpt.weight);
  ptr = parse(ptr, magOpt.param1);
  ptr = parse(ptr, magOpt.param2);
  ptr = parse(ptr, magOpt.param3);
  ptr = parse(ptr, magOpt.param4);
  ptr = parse(ptr, magOpt.param5);
  ptr = parse(ptr, magOpt.param6);
  ptr = parse(ptr, magOpt.param7);
  ptr = parse(ptr, magOpt.param8);
  ptr = parse(ptr, magOpt.param9);
  ptr = parse(ptr, magOpt.param10);
  ptr = parse(ptr, magOpt.param11);
  ptr = parse(ptr, magOpt.param12);
  ptr = parse(ptr, magOpt.param13);
  ptr = parse(ptr, magOpt.param14);
  ptr = parse(ptr, magOpt.param15);
  ptr = parse(ptr, magOpt.param16);
  ptr = parse(ptr, magOpt.excFunc1);
  ptr = parse(ptr, magOpt.excFunc2);
  ptr = parse(ptr, magOpt.excFunc3);
  ptr = parse(ptr, magOpt.excFunc4);
  ptr = parse(ptr, magOpt.excFunc5);
  ptr = parse(ptr, magOpt.excFunc6);
  ptr = parse(ptr, magOpt.availItemGroup1);
  ptr = parse(ptr, magOpt.reqClass1);
  ptr = parse(ptr, magOpt.availItemGroup2);
  ptr = parse(ptr, magOpt.reqClass2);
  ptr = parse(ptr, magOpt.availItemGroup3);
  ptr = parse(ptr, magOpt.reqClass3);
  ptr = parse(ptr, magOpt.availItemGroup4);
  ptr = parse(ptr, magOpt.reqClass4);
  ptr = parse(ptr, magOpt.availItemGroup5);
  ptr = parse(ptr, magOpt.reqClass5);
  ptr = parse(ptr, magOpt.availItemGroup6);
  ptr = parse(ptr, magOpt.reqClass6);
  ptr = parse(ptr, magOpt.availItemGroup7);
  ptr = parse(ptr, magOpt.reqClass7);
  ptr = parse(ptr, magOpt.availItemGroup8);
  ptr = parse(ptr, magOpt.reqClass8);
  ptr = parse(ptr, magOpt.availItemGroup9);
  ptr = parse(ptr, magOpt.reqClass9);
  ptr = parse(ptr, magOpt.availItemGroup10);
  parse(ptr, magOpt.reqClass10);
  return magOpt;
}

ref::Level parseLevelDataLine(const std::string &line) {
  ref::Level level;
  const char *ptr = line.data();
  ptr = parse(ptr, level.lvl);
  ptr = parse(ptr, level.exp_C);
  ptr = parse(ptr, level.exp_M);
  ptr = parse(ptr, level.cost_M);
  ptr = parse(ptr, level.cost_ST);
  ptr = parse(ptr, level.gust_Mob_Exp);
  ptr = parse(ptr, level.jobExp_Trader);
  ptr = parse(ptr, level.jobExp_Robber);
  parse(ptr, level.jobExp_Hunter);
  return level;
}

ref::Region parseRefRegionLine(const std::string &line) {
  ref::Region region;
  const char *ptr = line.data();
  ptr = parse(ptr, region.wRegionID);
  ptr = parse(ptr, region.x);
  ptr = parse(ptr, region.z);
  ptr = parse(ptr, region.continentName);
  ptr = parse(ptr, region.areaName);
  ptr = parse(ptr, region.isBattleField);
  ptr = parse(ptr, region.climate);
  ptr = parse(ptr, region.maxCapacity);
  ptr = parse(ptr, region.assocObjID);
  ptr = parse(ptr, region.assocServer);
  ptr = parse(ptr, region.assocFile256);
  ptr = parse(ptr, region.linkedRegion_1);
  ptr = parse(ptr, region.linkedRegion_2);
  ptr = parse(ptr, region.linkedRegion_3);
  ptr = parse(ptr, region.linkedRegion_4);
  ptr = parse(ptr, region.linkedRegion_5);
  ptr = parse(ptr, region.linkedRegion_6);
  ptr = parse(ptr, region.linkedRegion_7);
  ptr = parse(ptr, region.linkedRegion_8);
  ptr = parse(ptr, region.linkedRegion_9);
  parse(ptr, region.linkedRegion_10);
  return region;
}

ref::Skill parseSkilldataLine(const std::vector<absl::string_view> &linePieces) {
  ref::Skill skill;
  parse(linePieces[0], skill.service);
  parse(linePieces[1], skill.id);
  parse(linePieces[2], skill.groupId);
  parse(linePieces[3], skill.basicCode);
  parse(linePieces[4], skill.basicName);
  parse(linePieces[5], skill.basicGroup);
  parse(linePieces[6], skill.basicOriginal);
  parse(linePieces[7], skill.basicLevel);
  parse(linePieces[8], skill.basicActivity);
  parse(linePieces[9], skill.basicChainCode);
  parse(linePieces[10], skill.basicRecycleCost);
  parse(linePieces[11], skill.actionPreparingTime);
  parse(linePieces[12], skill.actionCastingTime);
  parse(linePieces[13], skill.actionActionDuration);
  parse(linePieces[14], skill.actionReuseDelay);
  parse(linePieces[15], skill.actionCoolTime);
  parse(linePieces[16], skill.actionFlyingSpeed);
  parse(linePieces[17], skill.actionInterruptable);
  parse(linePieces[18], skill.actionOverlap);
  parse(linePieces[19], skill.actionAutoAttackType);
  parse(linePieces[20], skill.actionInTown);
  parse(linePieces[21], skill.actionRange);
  parse(linePieces[22], skill.targetRequired);
  parse(linePieces[23], skill.targetTypeAnimal);
  parse(linePieces[24], skill.targetTypeLand);
  parse(linePieces[25], skill.targetTypeBuilding);
  parse(linePieces[26], skill.targetGroupSelf);
  parse(linePieces[27], skill.targetGroupAlly);
  parse(linePieces[28], skill.targetGroupParty);
  parse(linePieces[29], skill.targetGroupEnemy_M);
  parse(linePieces[30], skill.targetGroupEnemy_P);
  parse(linePieces[31], skill.targetGroupNeutral);
  parse(linePieces[32], skill.targetGroupDontCare);
  parse(linePieces[33], skill.targetEtcSelectDeadBody);
  parse(linePieces[34], skill.reqCommonMastery1);
  parse(linePieces[35], skill.reqCommonMastery2);
  parse(linePieces[36], skill.reqCommonMasteryLevel1);
  parse(linePieces[37], skill.reqCommonMasteryLevel2);
  parse(linePieces[38], skill.reqCommonStr);
  parse(linePieces[39], skill.reqCommonInt);
  parse(linePieces[40], skill.reqLearnSkill1);
  parse(linePieces[41], skill.reqLearnSkill2);
  parse(linePieces[42], skill.reqLearnSkill3);
  parse(linePieces[43], skill.reqLearnSkillLevel1);
  parse(linePieces[44], skill.reqLearnSkillLevel2);
  parse(linePieces[45], skill.reqLearnSkillLevel3);
  parse(linePieces[46], skill.reqLearnSP);
  parse(linePieces[47], skill.reqLearnRace);
  parse(linePieces[48], skill.reqRestriction1);
  parse(linePieces[49], skill.reqRestriction2);
  parse(linePieces[50], skill.reqCastWeapon1);
  parse(linePieces[51], skill.reqCastWeapon2);
  parse(linePieces[52], skill.consumeHP);
  parse(linePieces[53], skill.consumeMP);
  parse(linePieces[54], skill.consumeHPRatio);
  parse(linePieces[55], skill.consumeMPRatio);
  parse(linePieces[56], skill.consumeWHAN);
  parse(linePieces[57], skill.uiSkillTab);
  parse(linePieces[58], skill.uiSkillPage);
  parse(linePieces[59], skill.uiSkillColumn);
  parse(linePieces[60], skill.uiSkillRow);
  parse(linePieces[61], skill.uiIconFile);
  parse(linePieces[62], skill.uiSkillName);
  parse(linePieces[63], skill.uiSkillToolTip);
  parse(linePieces[64], skill.uiSkillToolTip_Desc);
  parse(linePieces[65], skill.uiSkillStudy_Desc);
  parse(linePieces[66], skill.aiAttackChance);
  parse(linePieces[67], skill.aiSkillType);
  parse(linePieces[68], skill.params[0]);
  parse(linePieces[69], skill.params[1]);
  parse(linePieces[70], skill.params[2]);
  parse(linePieces[71], skill.params[3]);
  parse(linePieces[72], skill.params[4]);
  parse(linePieces[73], skill.params[5]);
  parse(linePieces[74], skill.params[6]);
  parse(linePieces[75], skill.params[7]);
  parse(linePieces[76], skill.params[8]);
  parse(linePieces[77], skill.params[9]);
  parse(linePieces[78], skill.params[10]);
  parse(linePieces[79], skill.params[11]);
  parse(linePieces[80], skill.params[12]);
  parse(linePieces[81], skill.params[13]);
  parse(linePieces[82], skill.params[14]);
  parse(linePieces[83], skill.params[15]);
  parse(linePieces[84], skill.params[16]);
  parse(linePieces[85], skill.params[17]);
  parse(linePieces[86], skill.params[18]);
  parse(linePieces[87], skill.params[19]);
  parse(linePieces[88], skill.params[20]);
  parse(linePieces[89], skill.params[21]);
  parse(linePieces[90], skill.params[22]);
  parse(linePieces[91], skill.params[23]);
  parse(linePieces[92], skill.params[24]);
  parse(linePieces[93], skill.params[25]);
  parse(linePieces[94], skill.params[26]);
  parse(linePieces[95], skill.params[27]);
  parse(linePieces[96], skill.params[28]);
  parse(linePieces[97], skill.params[29]);
  parse(linePieces[98], skill.params[30]);
  parse(linePieces[99], skill.params[31]);
  parse(linePieces[100], skill.params[32]);
  parse(linePieces[101], skill.params[33]);
  parse(linePieces[102], skill.params[34]);
  parse(linePieces[103], skill.params[35]);
  parse(linePieces[104], skill.params[36]);
  parse(linePieces[105], skill.params[37]);
  parse(linePieces[106], skill.params[38]);
  parse(linePieces[107], skill.params[39]);
  parse(linePieces[108], skill.params[40]);
  parse(linePieces[109], skill.params[41]);
  parse(linePieces[110], skill.params[42]);
  parse(linePieces[111], skill.params[43]);
  parse(linePieces[112], skill.params[44]);
  parse(linePieces[113], skill.params[45]);
  parse(linePieces[114], skill.params[46]);
  parse(linePieces[115], skill.params[47]);
  parse(linePieces[116], skill.params[48]);
  parse(linePieces[117], skill.params[49]);
  return skill;
}

ref::Teleport parseTeleportbuildingLine(const std::string &line) {
  ref::Teleport teleportBuilding;
  const char *ptr = line.data();
  ptr = parse(ptr, teleportBuilding.service);
  ptr = parse(ptr, teleportBuilding.id);
  ptr = parse(ptr, teleportBuilding.codeName128);
  ptr = parse(ptr, teleportBuilding.objName128);
  ptr = parse(ptr, teleportBuilding.orgObjCodeName128);
  ptr = parse(ptr, teleportBuilding.nameStrID128);
  ptr = parse(ptr, teleportBuilding.descStrID128);
  ptr = parse(ptr, teleportBuilding.cashItem);
  ptr = parse(ptr, teleportBuilding.bionic);
  ptr = parse(ptr, teleportBuilding.typeId1);
  ptr = parse(ptr, teleportBuilding.typeId2);
  ptr = parse(ptr, teleportBuilding.typeId3);
  ptr = parse(ptr, teleportBuilding.typeId4);
  ptr = parse(ptr, teleportBuilding.decayTime);
  ptr = parse(ptr, teleportBuilding.country);
  ptr = parse(ptr, teleportBuilding.rarity);
  ptr = parse(ptr, teleportBuilding.canTrade);
  ptr = parse(ptr, teleportBuilding.canSell);
  ptr = parse(ptr, teleportBuilding.canBuy);
  ptr = parse(ptr, teleportBuilding.canBorrow);
  ptr = parse(ptr, teleportBuilding.canDrop);
  ptr = parse(ptr, teleportBuilding.canPick);
  ptr = parse(ptr, teleportBuilding.canRepair);
  ptr = parse(ptr, teleportBuilding.canRevive);
  ptr = parse(ptr, teleportBuilding.canUse);
  ptr = parse(ptr, teleportBuilding.canThrow);
  ptr = parse(ptr, teleportBuilding.price);
  ptr = parse(ptr, teleportBuilding.costRepair);
  ptr = parse(ptr, teleportBuilding.costRevive);
  ptr = parse(ptr, teleportBuilding.costBorrow);
  ptr = parse(ptr, teleportBuilding.keepingFee);
  ptr = parse(ptr, teleportBuilding.sellPrice);
  ptr = parse(ptr, teleportBuilding.reqLevelType1);
  ptr = parse(ptr, teleportBuilding.reqLevel1);
  ptr = parse(ptr, teleportBuilding.reqLevelType2);
  ptr = parse(ptr, teleportBuilding.reqLevel2);
  ptr = parse(ptr, teleportBuilding.reqLevelType3);
  ptr = parse(ptr, teleportBuilding.reqLevel3);
  ptr = parse(ptr, teleportBuilding.reqLevelType4);
  ptr = parse(ptr, teleportBuilding.reqLevel4);
  ptr = parse(ptr, teleportBuilding.maxContain);
  ptr = parse(ptr, teleportBuilding.regionID);
  ptr = parse(ptr, teleportBuilding.dir);
  ptr = parse(ptr, teleportBuilding.offsetX);
  ptr = parse(ptr, teleportBuilding.offsetY);
  ptr = parse(ptr, teleportBuilding.offsetZ);
  ptr = parse(ptr, teleportBuilding.speed1);
  ptr = parse(ptr, teleportBuilding.speed2);
  ptr = parse(ptr, teleportBuilding.scale);
  ptr = parse(ptr, teleportBuilding.bcHeight);
  ptr = parse(ptr, teleportBuilding.bcRadius);
  ptr = parse(ptr, teleportBuilding.eventID);
  ptr = parse(ptr, teleportBuilding.assocFileObj128);
  ptr = parse(ptr, teleportBuilding.assocFileDrop128);
  ptr = parse(ptr, teleportBuilding.assocFileIcon128);
  ptr = parse(ptr, teleportBuilding.assocFile1_128);
  ptr = parse(ptr, teleportBuilding.assocFile2_128);
  parse(ptr, teleportBuilding.link);
  return teleportBuilding;
}

ref::ScrapOfPackageItem parseScrapOfPackageItemLine(const std::string &line) {
  ref::ScrapOfPackageItem scrap;
  const char *ptr = line.data();
  ptr = parse(ptr, scrap.service);
  ptr = parse(ptr, scrap.country);
  ptr = parse(ptr, scrap.refPackageItemCodeName);
  ptr = parse(ptr, scrap.refItemCodeName);
  ptr = parse(ptr, scrap.optLevel);
  ptr = parse(ptr, scrap.variance);
  ptr = parse(ptr, scrap.data);
  ptr = parse(ptr, scrap.magParamNum);
  ptr = parse(ptr, scrap.magParams[0]);
  ptr = parse(ptr, scrap.magParams[1]);
  ptr = parse(ptr, scrap.magParams[2]);
  ptr = parse(ptr, scrap.magParams[3]);
  ptr = parse(ptr, scrap.magParams[4]);
  ptr = parse(ptr, scrap.magParams[5]);
  ptr = parse(ptr, scrap.magParams[6]);
  ptr = parse(ptr, scrap.magParams[7]);
  ptr = parse(ptr, scrap.magParams[8]);
  ptr = parse(ptr, scrap.magParams[9]);
  ptr = parse(ptr, scrap.magParams[10]);
  ptr = parse(ptr, scrap.magParams[11]);
  ptr = parse(ptr, scrap.param1);
  ptr = parse(ptr, scrap.param1Desc128);
  ptr = parse(ptr, scrap.param2);
  ptr = parse(ptr, scrap.param2Desc128);
  ptr = parse(ptr, scrap.param3);
  ptr = parse(ptr, scrap.param3Desc128);
  ptr = parse(ptr, scrap.param4);
  ptr = parse(ptr, scrap.param4Desc128);
  parse(ptr, scrap.index);
  return scrap;
}

ref::ShopTab parseShopTabLine(const std::string &line) {
  ref::ShopTab tab;
  const char *ptr = line.data();
  ptr = parse(ptr, tab.service);
  ptr = parse(ptr, tab.country);
  ptr = parse(ptr, tab.id);
  ptr = parse(ptr, tab.codeName128);
  ptr = parse(ptr, tab.refTabGroupCodeName);
  parse(ptr, tab.strID128Tab);
  return tab;
}

ref::ShopGroup parseShopGroupLine(const std::string &line) {
  ref::ShopGroup group;
  const char *ptr = line.data();
  ptr = parse(ptr, group.service);
  ptr = parse(ptr, group.country);
  ptr = parse(ptr, group.id);
  ptr = parse(ptr, group.codeName128);
  ptr = parse(ptr, group.refNPCCodeName);
  ptr = parse(ptr, group.param1);
  ptr = parse(ptr, group.param1Desc128);
  ptr = parse(ptr, group.param2);
  ptr = parse(ptr, group.param2Desc128);
  ptr = parse(ptr, group.param3);
  ptr = parse(ptr, group.param3Desc128);
  ptr = parse(ptr, group.param4);
  parse(ptr, group.param4Desc128);
  return group;
}

ref::ShopGood parseShopGoodLine(const std::string &line) {
  ref::ShopGood good;
  const char *ptr = line.data();
  ptr = parse(ptr, good.service);
  ptr = parse(ptr, good.country);
  ptr = parse(ptr, good.refTabCodeName);
  ptr = parse(ptr, good.refPackageItemCodeName);
  ptr = parse(ptr, good.slotIndex);
  ptr = parse(ptr, good.param1);
  ptr = parse(ptr, good.param1Desc128);
  ptr = parse(ptr, good.param2);
  ptr = parse(ptr, good.param2Desc128);
  ptr = parse(ptr, good.param3);
  ptr = parse(ptr, good.param3Desc128);
  ptr = parse(ptr, good.param4);
  parse(ptr, good.param4Desc128);
  return good;
}

ref::MappingShopGroup parseMappingShopGroupLine(const std::string &line) {
  ref::MappingShopGroup mapping;
  const char *ptr = line.data();
  ptr = parse(ptr, mapping.service);
  ptr = parse(ptr, mapping.country);
  ptr = parse(ptr, mapping.refShopGroupCodeName);
  parse(ptr, mapping.refShopCodeName);
  return mapping;
}

ref::MappingShopWithTab parseMappingShopWithTabLine(const std::string &line) {
  ref::MappingShopWithTab mapping;
  const char *ptr = line.data();
  ptr = parse(ptr, mapping.service);
  ptr = parse(ptr, mapping.country);
  ptr = parse(ptr, mapping.refShopCodeName);
  parse(ptr, mapping.refTabGroupCodeName);
  return mapping;
}

ref::TextZoneName parseTextZoneNameLine(const std::vector<absl::string_view> &linePieces) {
  ref::TextZoneName textZoneName;
  parse(linePieces[0], textZoneName.service);
  parse(linePieces[1], textZoneName.codeName128);
  parse(linePieces[2], textZoneName.korean);
  parse(linePieces[3], textZoneName.unkLang0);
  parse(linePieces[4], textZoneName.chineseTraditional);
  parse(linePieces[5], textZoneName.chineseSimplified);
  parse(linePieces[6], textZoneName.german);
  parse(linePieces[7], textZoneName.japanese);
  parse(linePieces[8], textZoneName.english);
  parse(linePieces[9], textZoneName.vietnamese);
  parse(linePieces[10], textZoneName.portuguese);
  parse(linePieces[11], textZoneName.russian);
  parse(linePieces[12], textZoneName.turkish);
  parse(linePieces[13], textZoneName.spanish);
  parse(linePieces[14], textZoneName.arabic);
  return textZoneName;
}

ref::Text parseTextLine(const std::vector<absl::string_view> &linePieces) {
  ref::Text text;
  parse(linePieces[0], text.service);
  parse(linePieces[1], text.codeName128);
  parse(linePieces[2], text.korean);
  parse(linePieces[3], text.unkLang0);
  parse(linePieces[4], text.chineseTraditional);
  parse(linePieces[5], text.chineseSimplified);
  parse(linePieces[6], text.german);
  parse(linePieces[7], text.japanese);
  parse(linePieces[8], text.english);
  parse(linePieces[9], text.vietnamese);
  parse(linePieces[10], text.portuguese);
  parse(linePieces[11], text.russian);
  parse(linePieces[12], text.turkish);
  parse(linePieces[13], text.spanish);
  parse(linePieces[14], text.arabic);
  return text;
}

ref::Mastery parseMasteryLine(const std::vector<absl::string_view> &linePieces) {
  ref::Mastery mastery;
  parse(linePieces[0], mastery.masteryId);
  parse(linePieces[1], mastery.masteryName);
  parse(linePieces[2], mastery.masteryNameCode);
  parse(linePieces[3], mastery.groupNum);
  parse(linePieces[4], mastery.masteryDescriptionId);
  parse(linePieces[5], mastery.tabNameCode);
  parse(linePieces[6], mastery.tabId);
  parse(linePieces[7], mastery.skillToolTipType);
  parse(linePieces[8], mastery.weaponType1);
  parse(linePieces[9], mastery.weaponType2);
  parse(linePieces[10], mastery.weaponType3);
  parse(linePieces[11], mastery.masteryIcon);
  parse(linePieces[12], mastery.masteryFocusIcon);
  return mastery;
}

uint16_t parseGatePort(const std::vector<uint8_t> &data) {
  // This file contains a fixed size (8) ascii string representing the gateway server port
  // Port most-significant-digit starts in pos 0, unused characters are nulls
  if (data.size() != 8) {
    throw std::runtime_error("Expecting GATEINFO.TXT to have 8 bytes");
  }
  int port = 0;
  int index = 0;
  while (index < data.size() && data[index] != 0) {
    port *= 10;
    port += data[index] - '0';
    ++index;
  }
  if (port > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error("GATEINFO.TXT parsed a port that does not fit in a uint16_t");
  }
  return static_cast<uint16_t>(port);
}

DivisionInfo parseDivisionInfo(const std::vector<uint8_t> &data) {
  DivisionInfo divisionInfo;
  int readIndex=0;
  divisionInfo.locale = get<uint8_t>(data, readIndex);
  auto divisionCount = get<uint8_t>(data, readIndex);
  for (int i=0; i<divisionCount; ++i) {
    Division division;
    auto divisionNameLength = get<uint32_t>(data, readIndex);
    for (int j=0; j<divisionNameLength; ++j) {
      char c = get<char>(data, readIndex);
      division.name += c;
    }
    ++readIndex; // throw away null terminator
    auto gatewayCount = get<uint8_t>(data, readIndex);
    for (int j=0; j<gatewayCount; ++j) {
      auto addressLength = get<uint32_t>(data, readIndex);
      std::string address;
      for (int k=0; k<addressLength; ++k) {
        char c = get<char>(data, readIndex);
        address += c;
      }
      ++readIndex; // throw away null terminator
      division.gatewayIpAddresses.push_back(address);
      divisionInfo.divisions.push_back(division);
    }
  }
  return divisionInfo;
}

std::vector<std::string> split(const std::string &str, const std::string &delim) {
  std::vector<std::string> result;
  size_t last = 0;
  size_t next = 0;
  while ((next = str.find(delim, last)) != std::string::npos) {
    std::string s = str.substr(last, next-last);
    if (s != "") {
      result.push_back(s);
    }
    last = next + delim.size();
  }
  std::string s = str.substr(last);
  if (s != "") {
    result.push_back(s);
  }
  return result;
}

std::vector<std::string> splitAndSelectFields(const std::string &str, const std::string &delim, const std::vector<int> &fields) {
  if (fields.empty()) {
    return {};
  }
  int fieldsIndex=0;
  std::vector<std::string> result;
  size_t last = 0;
  size_t next = 0;
  int fieldNum=0;
  while ((next = str.find(delim, last)) != std::string::npos) {
    std::string s = str.substr(last, next-last);
    if (s != "") {
      if (fields[fieldsIndex] == fieldNum) {
        result.push_back(s);
        ++fieldsIndex;
        if (fieldsIndex >= fields.size()) {
          return result;
        }
      }
    }
    last = next + delim.size();
    ++fieldNum;
  }
  std::string s = str.substr(last);
  if (s != "") {
    if (fields[fieldsIndex] == fieldNum) {
      result.push_back(s);
    }
  }
  return result;
}

namespace {
void copySubstring(const char *begin, char **end, std::string &dest) {
  const char *ptr = begin;
  while (*ptr != 0 && *ptr != '\t') {
    ++ptr;
  }
  absl::string_view sv(begin, ptr-begin);
  dest = std::string(sv.data(), sv.size());
  *end = const_cast<char*>(ptr);
}
} // namespace

template<>
const char* parse<std::string>(const char *begin, std::string &result) {
  char *end;
  copySubstring(begin, &end, result);
  return end+1;
}

template<>
const char* parse<uint8_t>(const char *begin, uint8_t &result) {
  char *end;
  result = strtol(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<int16_t>(const char *begin, int16_t &result) {
  char *end;
  result = strtol(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<int32_t>(const char *begin, int32_t &result) {
  char *end;
  result = strtol(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<int64_t>(const char *begin, int64_t &result) {
  char *end;
  result = strtoll(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<float>(const char *begin, float &result) {
  char *end;
  result = strtof(begin, &end);
  return end+1;
}

} // namespace sro::pk2::parsing