#ifndef PK2_TEXT_ITEM_AND_SKILL_DATA_HPP_
#define PK2_TEXT_ITEM_AND_SKILL_DATA_HPP_

#include "../../../common/pk2/ref/textItemOrSkill.hpp"

#include <map>
#include <optional>
#include <string>

namespace pk2 {

class TextItemAndSkillData {
public:
  void addItem(ref::TextItemOrSkill &&itemOrSkill);
  const std::string& getItemName(const std::string &nameStrID128) const;
  const std::string& getSkillName(const std::string &uiSkillName) const;
  std::optional<std::string> getItemNameIfExists(const std::string &nameStrID128) const;
  std::optional<std::string> getSkillNameIfExists(const std::string &uiSkillName) const;
private:
  std::map<std::string, std::string> itemNames_;
  std::map<std::string, std::string> skillNames_;
};

} // namespace pk2

#endif // PK2_TEXT_ITEM_AND_SKILL_DATA_HPP_