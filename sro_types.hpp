#ifndef SRO_TYPES_HPP_
#define SRO_TYPES_HPP_

namespace sro::types {

enum class EntityType {
  kSelf,
  kCharacter,
  kPlayerCharacter,
  kNonplayerCharacter,
  kMonster,
  kItem,
  kPortal
};

} // namespace sro::types

#endif // SRO_TYPES_HPP_