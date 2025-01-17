#ifndef PK2_MASTERY_DATA_HPP
#define PK2_MASTERY_DATA_HPP

#include <silkroad_lib/pk2/ref/mastery.h>

#include <unordered_map>

namespace pk2 {

class MasteryData {
public:
  using MasteryMap = std::unordered_map<sro::pk2::ref::MasteryId,sro::pk2::ref::Mastery>;
  void addMastery(sro::pk2::ref::Mastery &&mastery);
  const sro::pk2::ref::Mastery& getMasteryById(sro::pk2::ref::MasteryId id) const;
  sro::pk2::ref::MasteryId getMasteryIdByMasteryNameCode(std::string_view masteryNameCode) const;
private:
  MasteryMap masteries_;
};

} // namespace pk2

#endif // PK2_MASTERY_DATA_HPP