#ifndef STATE_ENTITY_HPP
#define STATE_ENTITY_HPP

#include "../packet/parsing/parsedPacket.hpp" // Object

#include <memory>
#include <mutex>
#include <vector>

namespace state {
  
class Entity {
public:
  using EntityId = uint32_t;
  void trackEntity(std::shared_ptr<packet::parsing::Object> obj);
  void stopTrackingEntity(EntityId gId);
  bool trackingEntity(EntityId gId) const;
  packet::parsing::Object* getEntity(EntityId gId);
  const packet::parsing::Object* getEntity(EntityId gId) const;
  size_t size() const;
  const std::map<EntityId, std::shared_ptr<packet::parsing::Object>>& getEntityMap() const;
private:
  mutable std::mutex entityMutex_;
  std::map<EntityId, std::shared_ptr<packet::parsing::Object>> entityMap_;
};

} // namespace state

#endif // STATE_ENTITY_HPP