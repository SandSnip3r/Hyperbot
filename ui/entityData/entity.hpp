#ifndef ENTITY_DATA_ENTITY_HPP_
#define ENTITY_DATA_ENTITY_HPP_

#include "common.hpp"

#include "ui_proto/entity.pb.h"

#include <silkroad_lib/entity.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <optional>

namespace entity_data {

class Entity {
public:
  Entity() = default;
  Entity(proto::entity::Entity entityData) : entityData(entityData) {}
  virtual ~Entity() = default;
  sro::scalar_types::EntityGlobalId globalId;
  proto::entity::Entity entityData;
};

class MobileEntity : public Entity {
public:
  using Entity::Entity;
  virtual ~MobileEntity() = default;
  std::optional<Movement> movement;
};

class Character : public MobileEntity {
public:
  using MobileEntity::MobileEntity;
  virtual ~Character() = default;
  sro::entity::LifeState lifeState;
};

} // namespace entity_data

#endif // ENTITY_DATA_ENTITY_HPP_