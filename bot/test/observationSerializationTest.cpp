#include "rl/observation.hpp"
#include "event/eventCode.hpp"

#include <gtest/gtest.h>
#include <sstream>

using namespace rl;

TEST(ObservationSerialization, RoundTrip) {
  Observation obs;
  obs.timestamp_ = std::chrono::steady_clock::time_point(std::chrono::milliseconds(12345));
  obs.eventCode_ = event::EventCode::kRlUiStartTraining;
  obs.ourCurrentHp_ = 100;
  obs.ourMaxHp_ = 150;
  obs.ourCurrentMp_ = 50;
  obs.ourMaxMp_ = 60;
  obs.weAreKnockedDown_ = true;
  obs.opponentCurrentHp_ = 80;
  obs.opponentMaxHp_ = 90;
  obs.opponentCurrentMp_ = 40;
  obs.opponentMaxMp_ = 45;
  obs.opponentIsKnockedDown_ = false;
  obs.hpPotionCount_ = 3;
  for (size_t i = 0; i < obs.remainingTimeOurBuffs_.size(); ++i) {
    obs.remainingTimeOurBuffs_[i] = static_cast<float>(i) / 10.0f;
    obs.remainingTimeOpponentBuffs_[i] = static_cast<float>(i + 1) / 10.0f;
  }
  for (size_t i = 0; i < obs.remainingTimeOurDebuffs_.size(); ++i) {
    obs.remainingTimeOurDebuffs_[i] = static_cast<float>(i) / 5.0f;
    obs.remainingTimeOpponentDebuffs_[i] = static_cast<float>(i + 1) / 5.0f;
  }
  for (size_t i = 0; i < obs.skillCooldowns_.size(); ++i) {
    obs.skillCooldowns_[i] = static_cast<float>(i) / 20.0f;
  }
  for (size_t i = 0; i < obs.itemCooldowns_.size(); ++i) {
    obs.itemCooldowns_[i] = static_cast<float>(i) / 2.0f;
  }

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  obs.saveToStream(ss);
  ss.seekg(0);
  Observation loaded = Observation::loadFromStream(ss);

  EXPECT_EQ(obs.timestamp_, loaded.timestamp_);
  EXPECT_EQ(obs.eventCode_, loaded.eventCode_);
  EXPECT_EQ(obs.ourCurrentHp_, loaded.ourCurrentHp_);
  EXPECT_EQ(obs.ourMaxHp_, loaded.ourMaxHp_);
  EXPECT_EQ(obs.ourCurrentMp_, loaded.ourCurrentMp_);
  EXPECT_EQ(obs.ourMaxMp_, loaded.ourMaxMp_);
  EXPECT_EQ(obs.weAreKnockedDown_, loaded.weAreKnockedDown_);
  EXPECT_EQ(obs.opponentCurrentHp_, loaded.opponentCurrentHp_);
  EXPECT_EQ(obs.opponentMaxHp_, loaded.opponentMaxHp_);
  EXPECT_EQ(obs.opponentCurrentMp_, loaded.opponentCurrentMp_);
  EXPECT_EQ(obs.opponentMaxMp_, loaded.opponentMaxMp_);
  EXPECT_EQ(obs.opponentIsKnockedDown_, loaded.opponentIsKnockedDown_);
  EXPECT_EQ(obs.hpPotionCount_, loaded.hpPotionCount_);
  for (size_t i = 0; i < obs.remainingTimeOurBuffs_.size(); ++i) {
    EXPECT_FLOAT_EQ(obs.remainingTimeOurBuffs_[i], loaded.remainingTimeOurBuffs_[i]);
    EXPECT_FLOAT_EQ(obs.remainingTimeOpponentBuffs_[i], loaded.remainingTimeOpponentBuffs_[i]);
  }
  for (size_t i = 0; i < obs.remainingTimeOurDebuffs_.size(); ++i) {
    EXPECT_FLOAT_EQ(obs.remainingTimeOurDebuffs_[i], loaded.remainingTimeOurDebuffs_[i]);
    EXPECT_FLOAT_EQ(obs.remainingTimeOpponentDebuffs_[i], loaded.remainingTimeOpponentDebuffs_[i]);
  }
  for (size_t i = 0; i < obs.skillCooldowns_.size(); ++i) {
    EXPECT_FLOAT_EQ(obs.skillCooldowns_[i], loaded.skillCooldowns_[i]);
  }
  for (size_t i = 0; i < obs.itemCooldowns_.size(); ++i) {
    EXPECT_FLOAT_EQ(obs.itemCooldowns_[i], loaded.itemCooldowns_[i]);
  }
}

