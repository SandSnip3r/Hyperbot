#ifndef PK2_TEXT_DATA_HPP_
#define PK2_TEXT_DATA_HPP_

#include "../../../common/pk2/ref/text.hpp"

#include <map>
#include <optional>
#include <string>

namespace pk2 {

class TextData {
public:
  void addItem(ref::Text &&text);
  const std::string& getItemName(const std::string &nameStrID128) const;
  const std::string& getSkillName(const std::string &uiSkillName) const;
  const std::string& getMasteryName(const std::string &masteryNameCode) const;
  std::optional<std::string> getItemNameIfExists(const std::string &nameStrID128) const;
  std::optional<std::string> getSkillNameIfExists(const std::string &uiSkillName) const;
  std::optional<std::string> getMasteryNameIfExists(const std::string &masteryNameCode) const;

  // Expecting names like Pacheon, Wizard, Rogue, Fire, etc. Case matters.
  std::optional<std::string> getMasteryNameCodeIfExists(const std::string &masteryName) const;
private:
  std::map<std::string, std::string> itemNames_;
  std::map<std::string, std::string> skillNames_;
  std::map<std::string, std::string> masteryNames_;
};

} // namespace pk2

#endif // PK2_TEXT_DATA_HPP_