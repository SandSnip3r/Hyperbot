#include "rl/replayBuffer.hpp"

#include <gtest/gtest.h>

#include <random>
#include <numeric>

using namespace rl;
using ::testing::Test;
using TransitionId = ReplayBuffer<int32_t>::TransitionId;

class ReplayBufferTest : public Test {
protected:
  std::mt19937 rng{12345};

  // Utility to fill the buffer with N dummy transitions.
  void fillBuffer(ReplayBuffer<int32_t> &buffer, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      buffer.addTransition(static_cast<int32_t>(i));
    }
  }
};

// Basic initialization checks.
TEST_F(ReplayBufferTest, InitializationProperties) {
  ReplayBuffer<int32_t> buf(10, /*alpha=*/0.25f, /*beta=*/0.75f, /*epsilon=*/1e-3f);
  EXPECT_EQ(buf.capacity(), 10u);
  EXPECT_EQ(buf.size(), 0u);
}

// addTransition returns unique IDs and increments size.
TEST_F(ReplayBufferTest, AddTransitionUniqueIdsAndSize) {
  ReplayBuffer<int32_t> buf(3, 1.0f, 1.0f, 1e-6f);
  std::set<TransitionId> ids;

  TransitionId id0 = buf.addTransition(100);
  EXPECT_EQ(buf.size(), 1u);
  ids.insert(id0);

  TransitionId id1 = buf.addTransition(200);
  EXPECT_EQ(buf.size(), 2u);
  ids.insert(id1);

  TransitionId id2 = buf.addTransition(300);
  EXPECT_EQ(buf.size(), 3u);
  ids.insert(id2);

  // Check that all IDs are unique.
  EXPECT_EQ(ids.size(), 3u);
}

// deleteTransition decrements size and frees up the slot for reuse.
TEST_F(ReplayBufferTest, DeleteAndReuse) {
  ReplayBuffer<int32_t> buf(3, 1.0f, 1.0f, 1e-6f);
  auto id0 = buf.addTransition(1), id1 = buf.addTransition(2), id2 = buf.addTransition(3);
  EXPECT_EQ(buf.size(), 3u);

  buf.deleteTransition(id1);
  EXPECT_EQ(buf.size(), 2u);

  auto id3 = buf.addTransition(4);
  EXPECT_EQ(buf.size(), 3u);
}

// With alpha=0 (uniform priorities), all sample‐weights should be 1.0.
TEST_F(ReplayBufferTest, SampleUniformWhenAlphaZero) {
  const size_t N = 5;
  ReplayBuffer<int32_t> buf(N, /*alpha=*/0.0f, /*beta=*/0.5f, /*epsilon=*/1e-3f);
  fillBuffer(buf, N);

  auto samples = buf.sample(3, rng);
  EXPECT_EQ(samples.size(), 3u);
  for (auto &res : samples) {
    EXPECT_LT(res.transitionId, N);
    EXPECT_FLOAT_EQ(res.weight, 1.0f);
  }
}

// With beta=0, importance‐sampling weights are all 1.0 regardless of priorities.
TEST_F(ReplayBufferTest, WeightOneWhenBetaZero) {
  const size_t N = 4;
  ReplayBuffer<int32_t> buf(N, /*alpha=*/1.0f, /*beta=*/0.0f, /*epsilon=*/1e-3f);
  fillBuffer(buf, N);

  // Bump some priorities to non‐uniform values
  std::vector<TransitionId> ids(N);
  std::iota(ids.begin(), ids.end(), 0u);
  std::vector<float> prios = {0.1f, 0.2f, 0.3f, 0.4f};
  buf.updatePriorities(ids, prios);

  auto samples = buf.sample(4, rng);
  for (auto &res : samples) {
    EXPECT_FLOAT_EQ(res.weight, 1.0f);
  }
}

// If epsilon=0 and you set a priority to exactly zero, that transition should never be sampled.
// And with beta=1, the remaining transition's weight should be 1/(N*P)=1/(2*1)=0.5.
TEST_F(ReplayBufferTest, ZeroPriorityEpsilonZeroSuppressesSampling) {
  ReplayBuffer<int32_t> buf(2, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/0.0f);
  auto id0 = buf.addTransition(10);
  auto id1 = buf.addTransition(20);

  // id0 → priority 0, id1 → priority 1
  buf.updatePriorities({id0, id1}, {0.0f, 1.0f});

  // Run a large number of samples to ensure id0 is never sampled.
  const int iterations = 1000;
  for (int i = 0; i < iterations; ++i) {
    auto samples = buf.sample(2, rng);
    EXPECT_EQ(samples.size(), 2u);
    for (auto &res : samples) {
      EXPECT_NE(res.transitionId, id0)
          << "Transition with zero priority was sampled at iteration " << i;
      EXPECT_NEAR(res.weight, 1.0f, 1e-6f);
    }
  }
}

// Sampling with the same RNG seed should be reproducible.
TEST_F(ReplayBufferTest, SamplingReproducibility) {
  ReplayBuffer<int32_t> buf(3, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-6f);
  fillBuffer(buf, 3);

  std::mt19937 r1(42), r2(42);
  auto s1 = buf.sample(3, r1);
  auto s2 = buf.sample(3, r2);

  ASSERT_EQ(s1.size(), s2.size());
  for (size_t i = 0; i < s1.size(); ++i) {
    EXPECT_EQ(s1[i].transitionId, s2[i].transitionId);
    EXPECT_FLOAT_EQ(s1[i].weight, s2[i].weight);
  }
}

// A buffer of capacity 1 should always return the single element with weight 1.0
TEST_F(ReplayBufferTest, SingleCapacityAlwaysSame) {
  ReplayBuffer<int32_t> buf(1, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-3f);
  auto id = buf.addTransition(999);

  std::mt19937 localRng(7);
  for (int i = 0; i < 5; ++i) {
    auto samp = buf.sample(1, localRng);
    EXPECT_EQ(samp.size(), 1u);
    EXPECT_EQ(samp[0].transitionId, id);
    EXPECT_FLOAT_EQ(samp[0].weight, 1.0f);
  }
}

// Asking for more samples than capacity should throw
TEST_F(ReplayBufferTest, SampleCountExceedsCapacityThrows) {
  const size_t CAP = 3;
  ReplayBuffer<int32_t> buf(CAP, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-6f);
  fillBuffer(buf, CAP);

  // requesting CAP+1 should trigger an exception
  EXPECT_ANY_THROW({
    buf.sample(static_cast<int>(CAP + 1), rng);
  });
}

// Delete everything and then reuse the buffer.
TEST_F(ReplayBufferTest, DeleteAllAndReuse) {
  const size_t N = 4;
  ReplayBuffer<int32_t> buf(N, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-6f);
  fillBuffer(buf, N);

  // Delete all transitions
  for (TransitionId id = 0; id < N; ++id) {
    buf.deleteTransition(id);
  }
  EXPECT_EQ(buf.size(), 0u);

  // Should still be able to add & sample
  auto newId = buf.addTransition(42);
  EXPECT_EQ(buf.size(), 1u);
  auto samples = buf.sample(1, rng);
  ASSERT_EQ(samples.size(), 1u);
  EXPECT_EQ(samples[0].transitionId, newId);
}

TEST_F(ReplayBufferTest, DeleteFromPartialBuffer) {
  const size_t CAP = 5;
  ReplayBuffer<int32_t> buf(CAP, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-6f);

  TransitionId id;
  // Add only 3 transitions (partial fill)
  for (size_t i = 0; i < 3; ++i) {
    id = buf.addTransition(static_cast<int32_t>(i * 10));
  }
  EXPECT_EQ(buf.size(), 3u);

  // Delete one transition (delete ID 1)
  buf.deleteTransition(id);
  EXPECT_EQ(buf.size(), 2u);

  // Ensure that the deleted transition is not sampled
  for (int i = 0; i < 10; ++i) {
    auto samples = buf.sample(2, rng);
    for (const auto &s : samples) {
      EXPECT_NE(s.transitionId, id);
    }
  }
}

// Deleting more than exists should throw.
TEST_F(ReplayBufferTest, DeleteMoreThanExistsThrows) {
  const size_t N = 3;
  ReplayBuffer<int32_t> buf(N, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-6f);
  fillBuffer(buf, N);

  // Delete all valid IDs
  for (TransitionId id = 0; id < N; ++id) {
    buf.deleteTransition(id);
  }
  EXPECT_EQ(buf.size(), 0u);

  // Deleting the same IDs again must fail
  for (TransitionId id = 0; id < N; ++id) {
    EXPECT_ANY_THROW(buf.deleteTransition(id));
  }
  // And out-of-range IDs too
  EXPECT_ANY_THROW(buf.deleteTransition(static_cast<TransitionId>(N + 5)));
}

// addTransition should throw when the buffer is already full.
TEST_F(ReplayBufferTest, AddTransitionThrowsWhenFull) {
  ReplayBuffer<int32_t> buf(2, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-6f);
  buf.addTransition(100);
  buf.addTransition(200);
  EXPECT_EQ(buf.size(), 2u);
  EXPECT_ANY_THROW(buf.addTransition(300));
}

// sample(0) should be a no-op (empty result, no throw).
TEST_F(ReplayBufferTest, SampleZeroCountReturnsEmpty) {
  ReplayBuffer<int32_t> buf(3, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-6f);
  fillBuffer(buf, 2);
  auto samples = buf.sample(0, rng);
  EXPECT_TRUE(samples.empty());
}

// updatePriorities with mismatched ids/priorities lengths must throw.
TEST_F(ReplayBufferTest, UpdatePrioritiesMismatchedLengthsThrows) {
  ReplayBuffer<int32_t> buf(3, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-6f);
  fillBuffer(buf, 3);
  // two ids, but only one priority value
  EXPECT_ANY_THROW(buf.updatePriorities({0,1}, {0.1f}));
}

// updatePriorities with an out-of-range id must throw.
TEST_F(ReplayBufferTest, UpdatePrioritiesIdOutOfRangeThrows) {
  ReplayBuffer<int32_t> buf(3, /*alpha=*/1.0f, /*beta=*/1.0f, /*epsilon=*/1e-6f);
  fillBuffer(buf, 3);
  // id ‘4’ is beyond capacity 3
  EXPECT_ANY_THROW(buf.updatePriorities({0,4}, {0.1f,0.2f}));
}

// Constructor rejects capacity=0 or negative epsilon.
TEST(ReplayBufferInvalidParams, CapacityZeroOrNegativeEpsilon) {
  // capacity zero
  EXPECT_ANY_THROW((ReplayBuffer<int32_t>(0, 0.5f, 0.5f, 1e-6f)));
  // negative epsilon
  EXPECT_THROW((ReplayBuffer<int32_t>(5, 0.5f, 0.5f, -0.1f)), std::invalid_argument);
}

// Constructor enforces alpha, beta ∈ [0,1]
TEST(ReplayBufferInvalidParams, AlphaBetaRange) {
  const size_t N = 5;
  const float eps = 1e-6f;

  // α < 0 or α > 1
  EXPECT_THROW((ReplayBuffer<int32_t>(N, -0.1f, 0.5f, eps)), std::invalid_argument);
  EXPECT_THROW((ReplayBuffer<int32_t>(N,  1.1f, 0.5f, eps)), std::invalid_argument);

  // β < 0 or β > 1
  EXPECT_THROW((ReplayBuffer<int32_t>(N, 0.5f, -0.2f, eps)), std::invalid_argument);
  EXPECT_THROW((ReplayBuffer<int32_t>(N, 0.5f,  1.2f, eps)), std::invalid_argument);

  // Boundary values are OK
  EXPECT_NO_THROW(ReplayBuffer<int32_t>(N, 0.0f, 0.0f, eps));
  EXPECT_NO_THROW(ReplayBuffer<int32_t>(N, 1.0f, 1.0f, eps));
}