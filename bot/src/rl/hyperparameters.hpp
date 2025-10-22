#ifndef RL_HYPERPARAMETERS_HPP_
#define RL_HYPERPARAMETERS_HPP_

namespace rl::hyperparameters {

static constexpr int kPastObservationStackSize{8};
static constexpr int kBatchSize{256};
static constexpr int kReplayBufferMinimumBeforeTraining{40'000};
static constexpr int kReplayBufferCapacity{200'000};
static constexpr int kTargetNetworkUpdateInterval{10'000};
static constexpr int kTrainStepCheckpointInterval{10'000};
static constexpr float kTargetNetworkPolyakTau{0.0004f};
static constexpr int kTargetNetworkPolyakUpdateInterval{16};
static constexpr bool kTargetNetworkPolyakEnabled{true};
static constexpr float kGamma{0.997f};
static constexpr float kLearningRate{1e-6f};
static constexpr float kDropoutRate{0.05f};
static constexpr float kPerAlpha{0.01f};
static constexpr float kPerBetaStart{0.4f};
static constexpr float kPerBetaEnd{1.0f};
static constexpr int kPerTrainStepCountAnneal{50'000};
static constexpr float kInitialEpsilon{1.0f};
static constexpr float kFinalEpsilon{0.01f};
static constexpr int kEpsilonStepCountAnneal{50'000};
static constexpr int kTdLookahead{4};

} // namespace rl::hyperparameters

#endif // RL_HYPERPARAMETERS_HPP_