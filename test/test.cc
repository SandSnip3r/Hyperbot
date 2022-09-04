#include <silkroad_lib/position.h>

#include <gtest/gtest.h>

TEST(Position, TwoDefaultConstructedAreSame) {
  sro::Position p1;
  volatile int stackSpacer[10000];
  sro::Position p2;
  EXPECT_EQ(p1, p2);
}

TEST(Position, ConstructorAlreadyNormalized) {
  constexpr sro::RegionId regionId{25000};
  constexpr float xOffset{123}, yOffset{234}, zOffset{345};
  sro::Position pos(regionId, xOffset, yOffset, zOffset);
  EXPECT_EQ(pos.regionId(), regionId);
  EXPECT_FALSE(pos.isDungeon());
  EXPECT_EQ(pos.xSector(), 168);
  EXPECT_EQ(pos.zSector(), 97);
  EXPECT_EQ(pos.xOffset(), xOffset);
  EXPECT_EQ(pos.yOffset(), yOffset);
  EXPECT_EQ(pos.zOffset(), zOffset);
  const auto gameCoordinate = pos.toGameCoordinate();
  EXPECT_EQ(gameCoordinate.x, 6348);
  EXPECT_EQ(gameCoordinate.y, 994);
}