#include "rl/modelInputs.hpp"

namespace rl::model_inputs {

const ObservationAndActionStorage::ObservationAndActionType& BatchedTrainingInput::getObservationAndAction(ObservationAndActionStorage::Id id, const ObservationAndActionStorage &storage) {
  // First look up the item in our map.
  auto it = observationStorageMap.find(id);
  if (it != observationStorageMap.end()) {
    return it->second;
  }
  // Item is not found in our map, get it from storage and store it in our map.
  auto [it2, inserted] = observationStorageMap.emplace(id, storage.getObservationAndAction(id));
  if (!inserted) {
    throw std::runtime_error("Failed to insert observation and action into map");
  }
  return it2->second;
}

} // namespace rl::model_inputs