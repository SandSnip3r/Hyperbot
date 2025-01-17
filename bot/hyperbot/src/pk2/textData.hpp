#ifndef PK2_TEXT_DATA_HPP_
#define PK2_TEXT_DATA_HPP_

#include <silkroad_lib/pk2/ref/text.h>

#include <absl/container/flat_hash_map.h>

#include <optional>
#include <string>

namespace pk2 {

class TextData {
public:
  void addItem(sro::pk2::ref::Text &&text);
  const std::string& getItemName(const std::string &nameStrID128) const;
  const std::string& getSkillName(const std::string &uiSkillName) const;
  const std::string& getMasteryName(const std::string &masteryNameCode) const;
  std::optional<std::string> getItemNameIfExists(const std::string &nameStrID128) const;
  std::optional<std::string> getSkillNameIfExists(const std::string &uiSkillName) const;
  std::optional<std::string> getMasteryNameIfExists(const std::string &masteryNameCode) const;

  // Expecting names like Pacheon, Wizard, Rogue, Fire, etc. Case matters.
  std::optional<std::string> getMasteryNameCodeIfExists(const std::string &masteryName) const;
private:
  using MapType = absl::flat_hash_map<std::string, std::string>;
  MapType itemNames_;
  MapType skillNames_;
  MapType masteryNames_;

  const std::string* privateGetItemName(const std::string &nameStrID128) const;
  const std::string* privateGetSkillName(const std::string &uiSkillName) const;
  const std::string* privateGetMasteryName(const std::string &masteryNameCode) const;
};

} // namespace pk2

#endif // PK2_TEXT_DATA_HPP_