#ifndef SRO_ENTITY_TYPES_H_
#define SRO_ENTITY_TYPES_H_

namespace sro::entity_types {

enum class EntityType {
  kSelf,
  kCharacter,
  kPlayerCharacter,
  kNonplayerCharacter,
  kMonster,
  kItem,
  kPortal
};

} // namespace sro::entity_types

#endif // SRO_ENTITY_TYPES_H_