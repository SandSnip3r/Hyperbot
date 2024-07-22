#include "textData.hpp"

#include <absl/algorithm/container.h>
#include <absl/log/log.h>
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

} // anonymous namespace

namespace pk2 {

void TextData::addItem(ref::Text &&text) {
  if (text.service == 0) {
    // Not in service, skipping
    return;
  }
  const std::string kItemPrefix{"SN_ITEM"};
  const std::string kSkillPrefix{"SN_SKILL"};
  const std::string kMasteryPrefix{"UIIT_STT_"};
  auto isMastery = [](std::string_view str){
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
  auto startsWithPrefix = [](const std::string &str, const std::string &kPrefix) -> bool {
    if (str.size() < kPrefix.size()) {
      return false;
    }
    for (uint32_t i=0; i<kPrefix.size(); ++i) {
      if (str[i] != kPrefix[i]) {
        return false;
      }
    }
    return true;
  };

  // We will want to filter out items which end with _DESC or _STUDY.
  auto isStudyOrDesc = [](const std::string &str) {
    if (str.size() >= 5) {
      if (str.at(str.size()-1) == 'C' &&
          str.at(str.size()-2) == 'S' &&
          str.at(str.size()-3) == 'E' &&
          str.at(str.size()-4) == 'D' &&
          str.at(str.size()-5) == '_') {
        return true;
      }
      if (str.size() >= 6) {
        if (str.at(str.size()-1) == 'Y' &&
            str.at(str.size()-2) == 'D' &&
            str.at(str.size()-3) == 'U' &&
            str.at(str.size()-4) == 'T' &&
            str.at(str.size()-5) == 'S' &&
            str.at(str.size()-6) == '_') {
          return true;
        }
      }
    }
    return false;
  };

  if (startsWithPrefix(text.key, kItemPrefix) && !isStudyOrDesc(text.key))  {
    itemNames_.emplace(text.key, removeCarriageReturnAndLineFeed(text.english));
  } else if (startsWithPrefix(text.key, kSkillPrefix) && !isStudyOrDesc(text.key)) {
    skillNames_.emplace(text.key, removeCarriageReturnAndLineFeed(text.english));
  } else if (isMastery(text.key)) {
    masteryNames_.emplace(text.key, removeCarriageReturnAndLineFeed(text.english));
  } else {
    // TODO: Other types can be added
    //  All possible types: SN_COS, SN_EU, SN_EVENT, SN_FORTRESS,
    //    SN_GATE, SN_INS, SN_ITEM, SN_JUPITER, SN_MK, SN_MOB,
    //    SN_MOV, SN_MSKILL, SN_NPC, SN_PACKAGE, SN_SKILL,
    //    SN_STORE, SN_STRUCTURE, SN_THEME, SN_TRADE, SN_ZONE
  }
}

const std::string& TextData::getItemName(const std::string &nameStrID128) const {
  const auto it = itemNames_.find(nameStrID128);
  if (it == itemNames_.end()) {
    throw std::runtime_error("Could not find name for item \""+nameStrID128+"\"");
  }
  return it->second;
}

const std::string& TextData::getSkillName(const std::string &uiSkillName) const {
  const auto it = skillNames_.find(uiSkillName);
  if (it == skillNames_.end()) {
    throw std::runtime_error("Could not find name for skill \""+uiSkillName+"\"");
  }
  return it->second;
}

const std::string& TextData::getMasteryName(const std::string &masteryNameCode) const {
  const auto it = masteryNames_.find(masteryNameCode);
  if (it == masteryNames_.end()) {
    throw std::runtime_error("Could not find name for mastery \""+masteryNameCode+"\"");
  }
  return it->second;
}


std::optional<std::string> TextData::getItemNameIfExists(const std::string &nameStrID128) const {
  const auto it = itemNames_.find(nameStrID128);
  if (it == itemNames_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> TextData::getSkillNameIfExists(const std::string &uiSkillName) const {
  const auto it = skillNames_.find(uiSkillName);
  if (it == skillNames_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> TextData::getMasteryNameIfExists(const std::string &masteryNameCode) const {
  const auto it = masteryNames_.find(masteryNameCode);
  if (it == masteryNames_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> TextData::getMasteryNameCodeIfExists(const std::string &masteryName) const {
  for (const auto &codeAndNamePair : masteryNames_) {
    if (codeAndNamePair.second == masteryName) {
      return codeAndNamePair.first;
    }
  }
  return std::nullopt;
}


} // namespace pk2