#ifndef RL_MODEL_INPUTS_HPP_
#define RL_MODEL_INPUTS_HPP_

#include "rl/observation.hpp"
#include "rl/observationAndActionStorage.hpp"

#include <vector>

namespace rl::model_inputs {

// It is the user's responsibility to ensure that the lifetimes of the Observations used here exceed the lifetime of this object.
struct ModelInputView {
  // Observations & actions are stacked so that the model can see history.
  // Older observations are at the front and newer observations are at the back.
  std::vector<const Observation*> pastObservationStack;
  std::vector<int> pastActionStack;
  const Observation *currentObservation;
};

struct BatchedTrainingInput {
  std::unordered_map<ObservationAndActionStorage::Id, ObservationAndActionStorage::ObservationAndActionType> observationStorageMap;
  const ObservationAndActionStorage::ObservationAndActionType& getObservationAndAction(ObservationAndActionStorage::Id id, const ObservationAndActionStorage &storage);

  std::vector<ModelInputView> oldModelInputViews;
  std::vector<int> actionsTaken;
  std::vector<bool> isTerminals;
  std::vector<float> rewards;
  std::vector<ModelInputView> newModelInputViews;
  std::vector<float> importanceSamplingWeights;
};

} // namespace rl::model_inputs

#endif // RL_MODEL_INPUTS_HPP_