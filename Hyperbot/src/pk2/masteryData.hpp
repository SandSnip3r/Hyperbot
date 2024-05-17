#ifndef PK2_MASTERY_DATA_HPP
#define PK2_MASTERY_DATA_HPP

#include "../../../common/pk2/ref/mastery.hpp"

#include <unordered_map>

namespace pk2 {

class MasteryData {
public:
  using MasteryMap = std::unordered_map<ref::MasteryId,ref::Mastery>;
  void addMastery(pk2::ref::Mastery &&mastery);
  const pk2::ref::Mastery& getMasteryById(pk2::ref::MasteryId id) const;
private:
  MasteryMap masteries_;
};

} // namespace pk2

#endif // PK2_MASTERY_DATA_HPP