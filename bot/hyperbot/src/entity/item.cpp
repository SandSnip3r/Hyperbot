#include "item.hpp"

#include <absl/log/log.h>

namespace entity {

void Item::removeOwnership() {
  ownerJId.reset();
  if (eventBroker_) {
    eventBroker_->publishEvent<event::EntityOwnershipRemoved>(globalId);
  } else {
    LOG(WARNING) << "Trying to publish EntityOwnershipRemoved, but don't have event broker";
  }
}

} // namespace entity