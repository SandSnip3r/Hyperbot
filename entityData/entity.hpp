#ifndef ENTITY_DATA_ENTITY_HPP_
#define ENTITY_DATA_ENTITY_HPP_

#include "common.hpp"

#include "proto/broadcast.pb.h"

#include <silkroad_lib/entity.h>
#include <silkroad_lib/scalar_types.h>

#include <optional>

namespace entity_data {

class Entity {
public:
  virtual ~Entity() = default;
  sro::scalar_types::EntityGlobalId globalId;
  broadcast::EntityType entityType;
};

class MobileEntity : public Entity {
public:
  virtual ~MobileEntity() = default;
  std::optional<Movement> movement;
};

class Character : public MobileEntity {
public:
  virtual ~Character() = default;
  sro::entity::LifeState lifeState;
};

} // namespace entity_data

#endif // ENTITY_DATA_ENTITY_HPP_