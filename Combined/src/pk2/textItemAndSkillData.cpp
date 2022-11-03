#include "textItemAndSkillData.hpp"

#include "logging.hpp"

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
    for (int i=0; i<kPrefix.size(); ++i) {
      if (str[i] != kPrefix[i]) {
        return false;
      }
    }
    return true;
  };
  if (startsWithPrefix(itemOrSkill.key, kItemPrefix))  {
    itemNames_.emplace(itemOrSkill.key, itemOrSkill.english);
  } else if (startsWithPrefix(itemOrSkill.key, kSkillPrefix)) {
    skillNames_.emplace(itemOrSkill.key, itemOrSkill.english);
  } else {
    // TODO: Other types can be added
    //  All possible types: SN_COS, SN_EU, SN_EVENT, SN_FORTRESS,
    //    SN_GATE, SN_INS, SN_ITEM, SN_JUPITER, SN_MK, SN_MOB,
    //    SN_MOV, SN_MSKILL, SN_NPC, SN_PACKAGE, SN_SKILL,
    //    SN_STORE, SN_STRUCTURE, SN_THEME, SN_TRADE, SN_ZONE
  }
}

const std::string& TextItemAndSkillData::getItemName(const std::string &nameStrId128) const {
  const auto it = itemNames_.find(nameStrId128);
  if (it == itemNames_.end()) {
    throw std::runtime_error("Could not find name for item \""+nameStrId128+"\"");
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

const std::string* TextItemAndSkillData::tryGetItemName(const std::string &nameStrId128) const {
  const auto it = itemNames_.find(nameStrId128);
  if (it == itemNames_.end()) {
    return nullptr;
  }
  return &(it->second);
}

const std::string* TextItemAndSkillData::tryGetSkillName(const std::string &uiSkillName) const {
  const auto it = skillNames_.find(uiSkillName);
  if (it == skillNames_.end()) {
    return nullptr;
  }
  return &(it->second);
}

} // namespace pk2