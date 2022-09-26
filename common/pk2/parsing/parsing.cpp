#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING true

#include "parsing.hpp"

#include <array>
#include <iostream>
#include <codecvt>
#include <sstream>

namespace pk2::parsing {

std::string fileDataToString(const std::vector<uint8_t> &data) {
  // TODO: This is UTF16-LE, Improve this function using a proper conversion
  //  This function takes roughly twice as long as the parsing function
  //  PK2Reader::getEntryData took 1ms
  //  This function took 526ms
  //  Parsing took 299ms
  // const auto size = data.size()-2;
  // std::u16string u16((size/2)+1, '\0');
  // std::memcpy(u16.data(), data.data()+2, size);
  // return std::wstring_convert<
  //     std::codecvt_utf8_utf16<char16_t>, char16_t>{}.to_bytes(u16);
  // // https://stackoverflow.com/a/56723923/1148866

  if (data.size()%2 != 0) {
    throw std::runtime_error("Data is not evenly sized");
  }
  std::string result;
  result.reserve((data.size()-2)/2);
  // Skipping first 2 bytes for BOM
  for (int i=2; i<data.size(); i+=2) {
    result += (char)data[i];
  }
  return result;
}

std::vector<std::string> fileDataToStringLines(const std::vector<uint8_t> &data) {
  std::vector<std::string> lines;

  // Convert data to std::wstring
  std::wstring wstr;
  wstr.resize(data.size()/2);
  std::memcpy(wstr.data(), data.data(), data.size());

  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> converter;

  std::wistringstream is16(wstr);
  is16.imbue(std::locale(is16.getloc(), new std::codecvt_utf16<wchar_t, 0x10ffff, std::consume_header>()));
  is16.seekg(1, std::ios::beg);
  std::wstring wline;
  std::string u8line;
  while (getline(is16, wline)) {
    if (wline.empty()) {
      continue;
    }
    u8line = converter.to_bytes(wline);
    if (u8line.back() == '\r') {
      u8line.pop_back();
    }
    lines.emplace_back(std::move(u8line));
  }
  return lines;
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
  constexpr int kDataCount = 16;
  return isValidLine(kDataCount, line);
}

pk2::ref::Character parseCharacterdataLine(const std::string &line) {
  pk2::ref::Character character;
  const char *ptr = line.data();
  ptr = parse(ptr, character.service);
  ptr = parse(ptr, character.id);
  ptr = parse(ptr, character.codeName128);
  ptr = parse(ptr, character.objName128);
  ptr = parse(ptr, character.orgObjCodeName128);
  ptr = parse(ptr, character.nameStrID128);
  ptr = parse(ptr, character.descStrID128);
  ptr = parse(ptr, character.cashItem);
  ptr = parse(ptr, character.bionic);
  ptr = parse(ptr, character.typeId1);
  ptr = parse(ptr, character.typeId2);
  ptr = parse(ptr, character.typeId3);
  ptr = parse(ptr, character.typeId4);
  ptr = parse(ptr, character.decayTime);
  ptr = parse(ptr, character.country);
  ptr = parse(ptr, character.rarity);
  ptr = parse(ptr, character.canTrade);
  ptr = parse(ptr, character.canSell);
  ptr = parse(ptr, character.canBuy);
  ptr = parse(ptr, character.canBorrow);
  ptr = parse(ptr, character.canDrop);
  ptr = parse(ptr, character.canPick);
  ptr = parse(ptr, character.canRepair);
  ptr = parse(ptr, character.canRevive);
  ptr = parse(ptr, character.canUse);
  ptr = parse(ptr, character.canThrow);
  ptr = parse(ptr, character.price);
  ptr = parse(ptr, character.costRepair);
  ptr = parse(ptr, character.costRevive);
  ptr = parse(ptr, character.costBorrow);
  ptr = parse(ptr, character.keepingFee);
  ptr = parse(ptr, character.sellPrice);
  ptr = parse(ptr, character.reqLevelType1);
  ptr = parse(ptr, character.reqLevel1);
  ptr = parse(ptr, character.reqLevelType2);
  ptr = parse(ptr, character.reqLevel2);
  ptr = parse(ptr, character.reqLevelType3);
  ptr = parse(ptr, character.reqLevel3);
  ptr = parse(ptr, character.reqLevelType4);
  ptr = parse(ptr, character.reqLevel4);
  ptr = parse(ptr, character.maxContain);
  ptr = parse(ptr, character.regionID);
  ptr = parse(ptr, character.dir);
  ptr = parse(ptr, character.offsetX);
  ptr = parse(ptr, character.offsetY);
  ptr = parse(ptr, character.offsetZ);
  ptr = parse(ptr, character.speed1);
  ptr = parse(ptr, character.speed2);
  ptr = parse(ptr, character.scale);
  ptr = parse(ptr, character.bCHeight);
  ptr = parse(ptr, character.bCRadius);
  ptr = parse(ptr, character.eventID);
  ptr = parse(ptr, character.assocFileObj128);
  ptr = parse(ptr, character.assocFileDrop128);
  ptr = parse(ptr, character.assocFileIcon128);
  ptr = parse(ptr, character.assocFile1_128);
  ptr = parse(ptr, character.assocFile2_128);
  ptr = parse(ptr, character.lvl);
  ptr = parse(ptr, character.charGender);
  ptr = parse(ptr, character.maxHP);
  ptr = parse(ptr, character.maxMP);
  ptr = parse(ptr, character.inventorySize);
  ptr = parse(ptr, character.canStore_TID1);
  ptr = parse(ptr, character.canStore_TID2);
  ptr = parse(ptr, character.canStore_TID3);
  ptr = parse(ptr, character.canStore_TID4);
  ptr = parse(ptr, character.canBeVehicle);
  ptr = parse(ptr, character.canControl);
  ptr = parse(ptr, character.damagePortion);
  ptr = parse(ptr, character.maxPassenger);
  ptr = parse(ptr, character.assocTactics);
  ptr = parse(ptr, character.pd);
  ptr = parse(ptr, character.md);
  ptr = parse(ptr, character.par);
  ptr = parse(ptr, character.mar);
  ptr = parse(ptr, character.er);
  ptr = parse(ptr, character.br);
  ptr = parse(ptr, character.hr);
  ptr = parse(ptr, character.chr);
  ptr = parse(ptr, character.expToGive);
  ptr = parse(ptr, character.creepType);
  ptr = parse(ptr, character.knockdown);
  ptr = parse(ptr, character.kO_RecoverTime);
  ptr = parse(ptr, character.defaultSkill_1);
  ptr = parse(ptr, character.defaultSkill_2);
  ptr = parse(ptr, character.defaultSkill_3);
  ptr = parse(ptr, character.defaultSkill_4);
  ptr = parse(ptr, character.defaultSkill_5);
  ptr = parse(ptr, character.defaultSkill_6);
  ptr = parse(ptr, character.defaultSkill_7);
  ptr = parse(ptr, character.defaultSkill_8);
  ptr = parse(ptr, character.defaultSkill_9);
  ptr = parse(ptr, character.defaultSkill_10);
  ptr = parse(ptr, character.textureType);
  ptr = parse(ptr, character.except_1);
  ptr = parse(ptr, character.except_2);
  ptr = parse(ptr, character.except_3);
  ptr = parse(ptr, character.except_4);
  ptr = parse(ptr, character.except_5);
  ptr = parse(ptr, character.except_6);
  ptr = parse(ptr, character.except_7);
  ptr = parse(ptr, character.except_8);
  ptr = parse(ptr, character.except_9);
  ptr = parse(ptr, character.except_10);
  return character;
}

pk2::ref::Item parseItemdataLine(const std::string &line) {
  pk2::ref::Item item;
  const char *ptr = line.data();
  ptr = parse(ptr, item.service);
  ptr = parse(ptr, item.id);
  ptr = parse(ptr, item.codeName128);
  ptr = parse(ptr, item.objName128);
  ptr = parse(ptr, item.orgObjCodeName128);
  ptr = parse(ptr, item.nameStrID128);
  ptr = parse(ptr, item.descStrID128);
  ptr = parse(ptr, item.cashItem);
  ptr = parse(ptr, item.bionic);
  ptr = parse(ptr, item.typeId1);
  ptr = parse(ptr, item.typeId2);
  ptr = parse(ptr, item.typeId3);
  ptr = parse(ptr, item.typeId4);
  ptr = parse(ptr, item.decayTime);
  ptr = parse(ptr, item.country);
  ptr = parse(ptr, item.rarity);
  ptr = parse(ptr, item.canTrade);
  ptr = parse(ptr, item.canSell);
  ptr = parse(ptr, item.canBuy);
  ptr = parse(ptr, item.canBorrow);
  ptr = parse(ptr, item.canDrop);
  ptr = parse(ptr, item.canPick);
  ptr = parse(ptr, item.canRepair);
  ptr = parse(ptr, item.canRevive);
  ptr = parse(ptr, item.canUse);
  ptr = parse(ptr, item.canThrow);
  ptr = parse(ptr, item.price);
  ptr = parse(ptr, item.costRepair);
  ptr = parse(ptr, item.costRevive);
  ptr = parse(ptr, item.costBorrow);
  ptr = parse(ptr, item.keepingFee);
  ptr = parse(ptr, item.sellPrice);
  ptr = parse(ptr, item.reqLevelType1);
  ptr = parse(ptr, item.reqLevel1);
  ptr = parse(ptr, item.reqLevelType2);
  ptr = parse(ptr, item.reqLevel2);
  ptr = parse(ptr, item.reqLevelType3);
  ptr = parse(ptr, item.reqLevel3);
  ptr = parse(ptr, item.reqLevelType4);
  ptr = parse(ptr, item.reqLevel4);
  ptr = parse(ptr, item.maxContain);
  ptr = parse(ptr, item.regionID);
  ptr = parse(ptr, item.dir);
  ptr = parse(ptr, item.offsetX);
  ptr = parse(ptr, item.offsetY);
  ptr = parse(ptr, item.offsetZ);
  ptr = parse(ptr, item.speed1);
  ptr = parse(ptr, item.speed2);
  ptr = parse(ptr, item.scale);
  ptr = parse(ptr, item.bCHeight);
  ptr = parse(ptr, item.bCRadius);
  ptr = parse(ptr, item.eventID);
  ptr = parse(ptr, item.assocFileObj128);
  ptr = parse(ptr, item.assocFileDrop128);
  ptr = parse(ptr, item.assocFileIcon128);
  ptr = parse(ptr, item.assocFile1_128);
  ptr = parse(ptr, item.assocFile2_128);
  ptr = parse(ptr, item.maxStack);
  ptr = parse(ptr, item.reqGender);
  ptr = parse(ptr, item.reqStr);
  ptr = parse(ptr, item.reqInt);
  ptr = parse(ptr, item.itemClass);
  ptr = parse(ptr, item.setID);
  ptr = parse(ptr, item.dur_L);
  ptr = parse(ptr, item.dur_U);
  ptr = parse(ptr, item.pd_L);
  ptr = parse(ptr, item.pd_U);
  ptr = parse(ptr, item.pdInc);
  ptr = parse(ptr, item.er_L);
  ptr = parse(ptr, item.er_U);
  ptr = parse(ptr, item.eRInc);
  ptr = parse(ptr, item.par_L);
  ptr = parse(ptr, item.par_U);
  ptr = parse(ptr, item.parInc);
  ptr = parse(ptr, item.br_L);
  ptr = parse(ptr, item.br_U);
  ptr = parse(ptr, item.md_L);
  ptr = parse(ptr, item.md_U);
  ptr = parse(ptr, item.mdInc);
  ptr = parse(ptr, item.mar_L);
  ptr = parse(ptr, item.mar_U);
  ptr = parse(ptr, item.marInc);
  ptr = parse(ptr, item.pdStr_L);
  ptr = parse(ptr, item.pdStr_U);
  ptr = parse(ptr, item.mdInt_L);
  ptr = parse(ptr, item.mdInt_U);
  ptr = parse(ptr, item.quivered);
  ptr = parse(ptr, item.ammo1_TID4);
  ptr = parse(ptr, item.ammo2_TID4);
  ptr = parse(ptr, item.ammo3_TID4);
  ptr = parse(ptr, item.ammo4_TID4);
  ptr = parse(ptr, item.ammo5_TID4);
  ptr = parse(ptr, item.speedClass);
  ptr = parse(ptr, item.twoHanded);
  ptr = parse(ptr, item.range);
  ptr = parse(ptr, item.pAttackMin_L);
  ptr = parse(ptr, item.pAttackMin_U);
  ptr = parse(ptr, item.pAttackMax_L);
  ptr = parse(ptr, item.pAttackMax_U);
  ptr = parse(ptr, item.pAttackInc);
  ptr = parse(ptr, item.mAttackMin_L);
  ptr = parse(ptr, item.mAttackMin_U);
  ptr = parse(ptr, item.mAttackMax_L);
  ptr = parse(ptr, item.mAttackMax_U);
  ptr = parse(ptr, item.mAttackInc);
  ptr = parse(ptr, item.paStrMin_L);
  ptr = parse(ptr, item.paStrMin_U);
  ptr = parse(ptr, item.paStrMax_L);
  ptr = parse(ptr, item.paStrMax_U);
  ptr = parse(ptr, item.maInt_Min_L);
  ptr = parse(ptr, item.maInt_Min_U);
  ptr = parse(ptr, item.maInt_Max_L);
  ptr = parse(ptr, item.maInt_Max_U);
  ptr = parse(ptr, item.hr_L);
  ptr = parse(ptr, item.hr_U);
  ptr = parse(ptr, item.hRInc);
  ptr = parse(ptr, item.cHR_L);
  ptr = parse(ptr, item.cHR_U);
  ptr = parse(ptr, item.param1);
  ptr = parse(ptr, item.desc1_128);
  ptr = parse(ptr, item.param2);
  ptr = parse(ptr, item.desc2_128);
  ptr = parse(ptr, item.param3);
  ptr = parse(ptr, item.desc3_128);
  ptr = parse(ptr, item.param4);
  ptr = parse(ptr, item.desc4_128);
  ptr = parse(ptr, item.param5);
  ptr = parse(ptr, item.desc5_128);
  ptr = parse(ptr, item.param6);
  ptr = parse(ptr, item.desc6_128);
  ptr = parse(ptr, item.param7);
  ptr = parse(ptr, item.desc7_128);
  ptr = parse(ptr, item.param8);
  ptr = parse(ptr, item.desc8_128);
  ptr = parse(ptr, item.param9);
  ptr = parse(ptr, item.desc9_128);
  ptr = parse(ptr, item.param10);
  ptr = parse(ptr, item.desc10_128);
  ptr = parse(ptr, item.param11);
  ptr = parse(ptr, item.desc11_128);
  ptr = parse(ptr, item.param12);
  ptr = parse(ptr, item.desc12_128);
  ptr = parse(ptr, item.param13);
  ptr = parse(ptr, item.desc13_128);
  ptr = parse(ptr, item.param14);
  ptr = parse(ptr, item.desc14_128);
  ptr = parse(ptr, item.param15);
  ptr = parse(ptr, item.desc15_128);
  ptr = parse(ptr, item.param16);
  ptr = parse(ptr, item.desc16_128);
  ptr = parse(ptr, item.param17);
  ptr = parse(ptr, item.desc17_128);
  ptr = parse(ptr, item.param18);
  ptr = parse(ptr, item.desc18_128);
  ptr = parse(ptr, item.param19);
  ptr = parse(ptr, item.desc19_128);
  ptr = parse(ptr, item.param20);
  ptr = parse(ptr, item.desc20_128);
  ptr = parse(ptr, item.maxMagicOptCount);
  parse(ptr, item.childItemCount);
  return item;
}

pk2::ref::MagicOption parseMagicOptionDataLine(const std::string &line) {
  pk2::ref::MagicOption magOpt;
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

pk2::ref::Level parseLevelDataLine(const std::string &line) {
  pk2::ref::Level level;
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

pk2::ref::Skill parseSkilldataLine(const std::string &line) {
  pk2::ref::Skill skill;
  const char *ptr = line.data();
  ptr = parse(ptr, skill.service);
  ptr = parse(ptr, skill.id);
  ptr = parse(ptr, skill.groupId);
  ptr = parse(ptr, skill.basicCode);
  ptr = parse(ptr, skill.basicName);
  ptr = parse(ptr, skill.basicGroup);
  ptr = parse(ptr, skill.basicOriginal);
  ptr = parse(ptr, skill.basicLevel);
  ptr = parse(ptr, skill.basicActivity);
  ptr = parse(ptr, skill.basicChainCode);
  ptr = parse(ptr, skill.basicRecycleCost);
  ptr = parse(ptr, skill.actionPreparingTime);
  ptr = parse(ptr, skill.actionCastingTime);
  ptr = parse(ptr, skill.actionActionDuration);
  ptr = parse(ptr, skill.actionReuseDelay);
  ptr = parse(ptr, skill.actionCoolTime);
  ptr = parse(ptr, skill.actionFlyingSpeed);
  ptr = parse(ptr, skill.actionInterruptable);
  ptr = parse(ptr, skill.actionOverlap);
  ptr = parse(ptr, skill.actionAutoAttackType);
  ptr = parse(ptr, skill.actionInTown);
  ptr = parse(ptr, skill.actionRange);
  ptr = parse(ptr, skill.targetRequired);
  ptr = parse(ptr, skill.targetTypeAnimal);
  ptr = parse(ptr, skill.targetTypeLand);
  ptr = parse(ptr, skill.targetTypeBuilding);
  ptr = parse(ptr, skill.targetGroupSelf);
  ptr = parse(ptr, skill.targetGroupAlly);
  ptr = parse(ptr, skill.targetGroupParty);
  ptr = parse(ptr, skill.targetGroupEnemy_M);
  ptr = parse(ptr, skill.targetGroupEnemy_P);
  ptr = parse(ptr, skill.targetGroupNeutral);
  ptr = parse(ptr, skill.targetGroupDontCare);
  ptr = parse(ptr, skill.targetEtcSelectDeadBody);
  ptr = parse(ptr, skill.reqCommonMastery1);
  ptr = parse(ptr, skill.reqCommonMastery2);
  ptr = parse(ptr, skill.reqCommonMasteryLevel1);
  ptr = parse(ptr, skill.reqCommonMasteryLevel2);
  ptr = parse(ptr, skill.reqCommonStr);
  ptr = parse(ptr, skill.reqCommonInt);
  ptr = parse(ptr, skill.reqLearnSkill1);
  ptr = parse(ptr, skill.reqLearnSkill2);
  ptr = parse(ptr, skill.reqLearnSkill3);
  ptr = parse(ptr, skill.reqLearnSkillLevel1);
  ptr = parse(ptr, skill.reqLearnSkillLevel2);
  ptr = parse(ptr, skill.reqLearnSkillLevel3);
  ptr = parse(ptr, skill.reqLearnSP);
  ptr = parse(ptr, skill.reqLearnRace);
  ptr = parse(ptr, skill.reqRestriction1);
  ptr = parse(ptr, skill.reqRestriction2);
  ptr = parse(ptr, skill.reqCastWeapon1);
  ptr = parse(ptr, skill.reqCastWeapon2);
  ptr = parse(ptr, skill.consumeHP);
  ptr = parse(ptr, skill.consumeMP);
  ptr = parse(ptr, skill.consumeHPRatio);
  ptr = parse(ptr, skill.consumeMPRatio);
  ptr = parse(ptr, skill.consumeWHAN);
  ptr = parse(ptr, skill.uiSkillTab);
  ptr = parse(ptr, skill.uiSkillPage);
  ptr = parse(ptr, skill.uiSkillColumn);
  ptr = parse(ptr, skill.uiSkillRow);
  ptr = parse(ptr, skill.uiIconFile);
  ptr = parse(ptr, skill.uiSkillName);
  ptr = parse(ptr, skill.uiSkillToolTip);
  ptr = parse(ptr, skill.uiSkillToolTip_Desc);
  ptr = parse(ptr, skill.uiSkillStudy_Desc);
  ptr = parse(ptr, skill.aiAttackChance);
  ptr = parse(ptr, skill.aiSkillType);
  ptr = parse(ptr, skill.params[0]);
  ptr = parse(ptr, skill.params[1]);
  ptr = parse(ptr, skill.params[2]);
  ptr = parse(ptr, skill.params[3]);
  ptr = parse(ptr, skill.params[4]);
  ptr = parse(ptr, skill.params[5]);
  ptr = parse(ptr, skill.params[6]);
  ptr = parse(ptr, skill.params[7]);
  ptr = parse(ptr, skill.params[8]);
  ptr = parse(ptr, skill.params[9]);
  ptr = parse(ptr, skill.params[10]);
  ptr = parse(ptr, skill.params[11]);
  ptr = parse(ptr, skill.params[12]);
  ptr = parse(ptr, skill.params[13]);
  ptr = parse(ptr, skill.params[14]);
  ptr = parse(ptr, skill.params[15]);
  ptr = parse(ptr, skill.params[16]);
  ptr = parse(ptr, skill.params[17]);
  ptr = parse(ptr, skill.params[18]);
  ptr = parse(ptr, skill.params[19]);
  ptr = parse(ptr, skill.params[20]);
  ptr = parse(ptr, skill.params[21]);
  ptr = parse(ptr, skill.params[22]);
  ptr = parse(ptr, skill.params[23]);
  ptr = parse(ptr, skill.params[24]);
  ptr = parse(ptr, skill.params[25]);
  ptr = parse(ptr, skill.params[26]);
  ptr = parse(ptr, skill.params[27]);
  ptr = parse(ptr, skill.params[28]);
  ptr = parse(ptr, skill.params[29]);
  ptr = parse(ptr, skill.params[30]);
  ptr = parse(ptr, skill.params[31]);
  ptr = parse(ptr, skill.params[32]);
  ptr = parse(ptr, skill.params[33]);
  ptr = parse(ptr, skill.params[34]);
  ptr = parse(ptr, skill.params[35]);
  ptr = parse(ptr, skill.params[36]);
  ptr = parse(ptr, skill.params[37]);
  ptr = parse(ptr, skill.params[38]);
  ptr = parse(ptr, skill.params[39]);
  ptr = parse(ptr, skill.params[40]);
  ptr = parse(ptr, skill.params[41]);
  ptr = parse(ptr, skill.params[42]);
  ptr = parse(ptr, skill.params[43]);
  ptr = parse(ptr, skill.params[44]);
  ptr = parse(ptr, skill.params[45]);
  ptr = parse(ptr, skill.params[46]);
  ptr = parse(ptr, skill.params[47]);
  ptr = parse(ptr, skill.params[48]);
  parse(ptr, skill.params[49]);
  return skill;
}

pk2::ref::Teleport parseTeleportbuildingLine(const std::string &line) {
  pk2::ref::Teleport teleportBuilding;
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

pk2::ref::ScrapOfPackageItem parseScrapOfPackageItemLine(const std::string &line) {
  pk2::ref::ScrapOfPackageItem scrap;
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

pk2::ref::ShopTab parseShopTabLine(const std::string &line) {
  pk2::ref::ShopTab tab;
  const char *ptr = line.data();
  ptr = parse(ptr, tab.service);
  ptr = parse(ptr, tab.country);
  ptr = parse(ptr, tab.id);
  ptr = parse(ptr, tab.codeName128);
  ptr = parse(ptr, tab.refTabGroupCodeName);
  parse(ptr, tab.strID128Tab);
  return tab;
}

pk2::ref::ShopGroup parseShopGroupLine(const std::string &line) {
  pk2::ref::ShopGroup group;
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

pk2::ref::ShopGood parseShopGoodLine(const std::string &line) {
  pk2::ref::ShopGood good;
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

pk2::ref::MappingShopGroup parseMappingShopGroupLine(const std::string &line) {
  pk2::ref::MappingShopGroup mapping;
  const char *ptr = line.data();
  ptr = parse(ptr, mapping.service);
  ptr = parse(ptr, mapping.country);
  ptr = parse(ptr, mapping.refShopGroupCodeName);
  parse(ptr, mapping.refShopCodeName);
  return mapping;
}

pk2::ref::MappingShopWithTab parseMappingShopWithTabLine(const std::string &line) {
  pk2::ref::MappingShopWithTab mapping;
  const char *ptr = line.data();
  ptr = parse(ptr, mapping.service);
  ptr = parse(ptr, mapping.country);
  ptr = parse(ptr, mapping.refShopCodeName);
  parse(ptr, mapping.refTabGroupCodeName);
  return mapping;
}

pk2::ref::TextZoneName parseTextZoneNameLine(const std::string &line) {
  pk2::ref::TextZoneName textZoneName;
  const char *ptr = line.data();
  ptr = parse(ptr, textZoneName.service);
  ptr = parse(ptr, textZoneName.key);
  ptr = parse(ptr, textZoneName.korean);
  ptr = parse(ptr, textZoneName.unkLang0);
  ptr = parse(ptr, textZoneName.unkLang1);
  ptr = parse(ptr, textZoneName.unkLang2);
  ptr = parse(ptr, textZoneName.unkLang3);
  ptr = parse(ptr, textZoneName.unkLang4);
  ptr = parse(ptr, textZoneName.english);
  ptr = parse(ptr, textZoneName.vietnamese);
  ptr = parse(ptr, textZoneName.unkLang5);
  ptr = parse(ptr, textZoneName.unkLang6);
  ptr = parse(ptr, textZoneName.unkLang7);
  ptr = parse(ptr, textZoneName.unkLang8);
  ptr = parse(ptr, textZoneName.unkLang9);
  parse(ptr, textZoneName.unkLang10);
  return textZoneName;
}

pk2::ref::TextItemOrSkill parseTextItemOrSkillLine(const std::string &line) {
  pk2::ref::TextItemOrSkill textItemOrSkill;
  const char *ptr = line.data();
  ptr = parse(ptr, textItemOrSkill.service);
  ptr = parse(ptr, textItemOrSkill.key);
  ptr = parse(ptr, textItemOrSkill.korean);
  ptr = parse(ptr, textItemOrSkill.unkLang0);
  ptr = parse(ptr, textItemOrSkill.unkLang1);
  ptr = parse(ptr, textItemOrSkill.unkLang2);
  ptr = parse(ptr, textItemOrSkill.unkLang3);
  ptr = parse(ptr, textItemOrSkill.unkLang4);
  ptr = parse(ptr, textItemOrSkill.english);
  ptr = parse(ptr, textItemOrSkill.vietnamese);
  ptr = parse(ptr, textItemOrSkill.unkLang5);
  ptr = parse(ptr, textItemOrSkill.unkLang6);
  ptr = parse(ptr, textItemOrSkill.unkLang7);
  ptr = parse(ptr, textItemOrSkill.unkLang8);
  ptr = parse(ptr, textItemOrSkill.unkLang9);
  parse(ptr, textItemOrSkill.unkLang10);
  return textItemOrSkill;
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
  std::string_view sv(begin, ptr-begin);
  dest = sv;
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

} // namespace pk2::parsing