#include "textItemAndSkillData.hpp"

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

void TextItemAndSkillData::addItem(ref::TextItemOrSkill &&itemOrSkill) {
  if (itemOrSkill.service == 0) {
    // Not in service, skipping
    return;
  }
  const std::string kItemPrefix{"SN_ITEM"};
  const std::string kSkillPrefix{"SN_SKILL"};
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

  if (startsWithPrefix(itemOrSkill.key, kItemPrefix) && !isStudyOrDesc(itemOrSkill.key))  {
    itemNames_.emplace(itemOrSkill.key, removeCarriageReturnAndLineFeed(itemOrSkill.english));
  } else if (startsWithPrefix(itemOrSkill.key, kSkillPrefix) && !isStudyOrDesc(itemOrSkill.key)) {
    skillNames_.emplace(itemOrSkill.key, removeCarriageReturnAndLineFeed(itemOrSkill.english));
  } else {
    // TODO: Other types can be added
    //  All possible types: SN_COS, SN_EU, SN_EVENT, SN_FORTRESS,
    //    SN_GATE, SN_INS, SN_ITEM, SN_JUPITER, SN_MK, SN_MOB,
    //    SN_MOV, SN_MSKILL, SN_NPC, SN_PACKAGE, SN_SKILL,
    //    SN_STORE, SN_STRUCTURE, SN_THEME, SN_TRADE, SN_ZONE
  }
}

const std::string& TextItemAndSkillData::getItemName(const std::string &nameStrID128) const {
  const auto it = itemNames_.find(nameStrID128);
  if (it == itemNames_.end()) {
    throw std::runtime_error("Could not find name for item \""+nameStrID128+"\"");
  }
  return it->second;
}

const std::string& TextItemAndSkillData::getSkillName(const std::string &uiSkillName) const {
  const auto it = skillNames_.find(uiSkillName);
  if (it == skillNames_.end()) {
    throw std::runtime_error("Could not find name for skill \""+uiSkillName+"\"");
  }
  return it->second;
}

std::optional<std::string> TextItemAndSkillData::getItemNameIfExists(const std::string &nameStrID128) const {
  const auto it = itemNames_.find(nameStrID128);
  if (it == itemNames_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> TextItemAndSkillData::getSkillNameIfExists(const std::string &uiSkillName) const {
  const auto it = skillNames_.find(uiSkillName);
  if (it == skillNames_.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace pk2