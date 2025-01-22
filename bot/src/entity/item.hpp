#ifndef ENTITY_ITEM_HPP_
#define ENTITY_ITEM_HPP_

#include "entity.hpp"

#include <silkroad_lib/entity.hpp>

#include <cstdint>
#include <optional>

namespace entity {

class Item : public Entity {
public:
  sro::entity::ItemRarity rarity;
  std::optional<uint32_t> ownerJId;
  void removeOwnership();
  EntityType entityType() const override { return EntityType::kItem; }
};

} // namespace entity

#endif // ENTITY_ITEM_HPP_