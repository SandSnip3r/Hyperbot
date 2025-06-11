#include "textData.hpp"

#include <absl/algorithm/container.h>
#include <absl/log/log.h>
#include <absl/strings/match.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

namespace {

std::string removeCarriageReturnAndLineFeed(std::string str) {
  uint32_t charIndex=0;
  while (charIndex < str.size()) {
    if (str.at(charIndex) == '\r' ||
        str.at(charIndex) == '\n') {
      str.erase(charIndex);
    } else {
      // Valid character
      ++charIndex;
    }
  }
  return str;
}

const std::string kItemPrefix{"SN_ITEM"};

} // anonymous namespace

namespace sro::pk2 {

void TextData::addItem(sro::pk2::ref::Text &&text) {
  if (text.service == 0) {
    // Not in service, skipping
    return;
  }
  const std::string kSkillPrefix{"SN_SKILL"};
  const std::string kMasteryPrefix{"UIIT_STT_"};
  auto isMastery = [](std::string_view str) {
    static constexpr std::array arr = {
      "UIIT_STT_WARRIOR",
      "UIIT_STT_ROG",
      "UIIT_STT_WIZARD",
      "UIIT_STT_WARLOCK",
      "UIIT_STT_BARD",
      "UIIT_STT_CLERIC",
      "UIIT_STT_MASTERY_VI",
      "UIIT_STT_MASTERY_HEUK",
      "UIIT_STT_MASTERY_PA",
      "UIIT_STT_MASTERY_HAN",
      "UIIT_STT_MASTERY_PUNG",
      "UIIT_STT_MASTERY_HWA",
      "UIIT_STT_MASTERY_GI"
    };
    return absl::c_linear_search(arr, str);
  };

  // We will want to filter out items which end with _DESC or _STUDY.
  auto isStudyOrDesc = [](const std::string &str) {
    return (absl::EndsWith(str, "_DESC") ||
            absl::EndsWith(str, "_STUDY"));
  };

  if (absl::StartsWith(text.codeName128, kItemPrefix) && !isStudyOrDesc(text.codeName128)) {
    itemNames_.emplace(text.codeName128, removeCarriageReturnAndLineFeed(text.english));
  } else if (absl::StartsWith(text.codeName128, kSkillPrefix) && !isStudyOrDesc(text.codeName128)) {
    skillNames_.emplace(text.codeName128, removeCarriageReturnAndLineFeed(text.english));
  } else if (isMastery(text.codeName128)) {
    masteryNames_.emplace(text.codeName128, removeCarriageReturnAndLineFeed(text.english));
  } else {
    // TODO: Other types can be added
    //  All possible types: SN_COS, SN_EU, SN_EVENT, SN_FORTRESS,
    //    SN_GATE, SN_INS, SN_ITEM, SN_JUPITER, SN_MK, SN_MOB,
    //    SN_MOV, SN_MSKILL, SN_NPC, SN_PACKAGE, SN_SKILL,
    //    SN_STORE, SN_STRUCTURE, SN_THEME, SN_TRADE, SN_ZONE
  }
}

const std::string& TextData::getItemName(const std::string &nameStrID128) const {
  const std::string *itemName = privateGetItemName(nameStrID128);
  if (itemName == nullptr) {
    throw std::runtime_error(absl::StrFormat("Could not find name for item \"%s\"", nameStrID128));
  }
  return *itemName;
}

const std::string& TextData::getSkillName(const std::string &uiSkillName) const {
  const std::string *skillName = privateGetSkillName(uiSkillName);
  if (skillName == nullptr) {
    throw std::runtime_error(absl::StrFormat("Could not find name for skill \"%s\"", uiSkillName));
  }
  return *skillName;
}

const std::string& TextData::getMasteryName(const std::string &masteryNameCode) const {
  const std::string *masteryName = privateGetMasteryName(masteryNameCode);
  if (masteryName == nullptr) {
    throw std::runtime_error(absl::StrFormat("Could not find name for mastery \"%s\"", masteryNameCode));
  }
  return *masteryName;
}

std::optional<std::string> TextData::getItemNameIfExists(const std::string &nameStrID128) const {
  const std::string *name = privateGetItemName(nameStrID128);
  if (name == nullptr) {
    return std::nullopt;
  }
  return *name;
}

std::optional<std::string> TextData::getSkillNameIfExists(const std::string &uiSkillName) const {
  const std::string *name = privateGetSkillName(uiSkillName);
  if (name == nullptr) {
    return std::nullopt;
  }
  return *name;
}

std::optional<std::string> TextData::getMasteryNameIfExists(const std::string &masteryNameCode) const {
  const std::string *name = privateGetMasteryName(masteryNameCode);
  if (name == nullptr) {
    return std::nullopt;
  }
  return *name;
}

std::optional<std::string> TextData::getMasteryNameCodeIfExists(const std::string &masteryName) const {
  for (const auto &codeAndNamePair : masteryNames_) {
    if (codeAndNamePair.second == masteryName) {
      return codeAndNamePair.first;
    }
  }
  return std::nullopt;
}

const std::string* TextData::privateGetItemName(const std::string &nameStrID128) const {
  const auto it = itemNames_.find(nameStrID128);
  if (it == itemNames_.end()) {
    return nullptr;
  }
  return &(it->second);
}

const std::string* TextData::privateGetSkillName(const std::string &uiSkillName) const {
  const MapType *nameMap;
  if (absl::StartsWith(uiSkillName, kItemPrefix)) {
    // A skill might be from an item, for these, look up the name in the item name map. An example is a scroll.
    nameMap = &itemNames_;
  } else {
    nameMap = &skillNames_;
  }
  const auto it = nameMap->find(uiSkillName);
  if (it == nameMap->end()) {
    return nullptr;
  }
  return &(it->second);

}
const std::string* TextData::privateGetMasteryName(const std::string &masteryNameCode) const {
  const auto it = masteryNames_.find(masteryNameCode);
  if (it == masteryNames_.end()) {
    return nullptr;
  }
  return &(it->second);

}

} // namespace sro::pk2
