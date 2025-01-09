#ifndef ENTITY_PORTAL_HPP_
#define ENTITY_PORTAL_HPP_

#include "entity.hpp"

#include <cstdint>

namespace entity {

class Portal : public Entity {
public:
  uint8_t unkByte3;
  EntityType entityType() const override { return EntityType::kPortal; }
};

} // namespace entity

#endif // ENTITY_PORTAL_HPP_