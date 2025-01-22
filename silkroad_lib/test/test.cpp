#include <silkroad_lib/position.hpp>

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

TEST(Position, ConstructorRegionToTheRight) {
  // Initialize a position with an xOffset that extends over into the next region
  constexpr float xOffset{2043}, yOffset{234}, zOffset{345};
  sro::Position pos(25000, xOffset, yOffset, zOffset);
  EXPECT_EQ(pos.regionId(), 25001);
  EXPECT_FALSE(pos.isDungeon());
  EXPECT_EQ(pos.xSector(), 169);
  EXPECT_EQ(pos.zSector(), 97);
  EXPECT_EQ(pos.xOffset(), 123);
  EXPECT_EQ(pos.yOffset(), yOffset);
  EXPECT_EQ(pos.zOffset(), zOffset);
  const auto gameCoordinate = pos.toGameCoordinate();
  EXPECT_EQ(gameCoordinate.x, 6540);
  EXPECT_EQ(gameCoordinate.y, 994);
}

TEST(Position, ConstructorRegionToTheLeft) {
  // Initialize a position with an xOffset that extends over into the next region
  constexpr float xOffset{-1797}, yOffset{234}, zOffset{345};
  sro::Position pos(25000, xOffset, yOffset, zOffset);
  EXPECT_EQ(pos.regionId(), 24999);
  EXPECT_FALSE(pos.isDungeon());
  EXPECT_EQ(pos.xSector(), 167);
  EXPECT_EQ(pos.zSector(), 97);
  EXPECT_EQ(pos.xOffset(), 123);
  EXPECT_EQ(pos.yOffset(), yOffset);
  EXPECT_EQ(pos.zOffset(), zOffset);
  const auto gameCoordinate = pos.toGameCoordinate();
  EXPECT_EQ(gameCoordinate.x, 6156);
  EXPECT_EQ(gameCoordinate.y, 994);
}

TEST(Position, ConstructorRegionAbove) {
  // Initialize a position with an yOffset that extends over into the next region
  constexpr float xOffset{123}, yOffset{234}, zOffset{2265};
  sro::Position pos(25000, xOffset, yOffset, zOffset);
  EXPECT_EQ(pos.regionId(), 25256);
  EXPECT_FALSE(pos.isDungeon());
  EXPECT_EQ(pos.xSector(), 168);
  EXPECT_EQ(pos.zSector(), 98);
  EXPECT_EQ(pos.xOffset(), xOffset);
  EXPECT_EQ(pos.yOffset(), yOffset);
  EXPECT_EQ(pos.zOffset(), 345);
  const auto gameCoordinate = pos.toGameCoordinate();
  EXPECT_EQ(gameCoordinate.x, 6348);
  EXPECT_EQ(gameCoordinate.y, 1186);
}

TEST(Position, ConstructorRegionBelow) {
  // Initialize a position with an yOffset that extends over into the next region
  constexpr float xOffset{123}, yOffset{234}, zOffset{-1575};
  sro::Position pos(25000, xOffset, yOffset, zOffset);
  EXPECT_EQ(pos.regionId(), 24744);
  EXPECT_FALSE(pos.isDungeon());
  EXPECT_EQ(pos.xSector(), 168);
  EXPECT_EQ(pos.zSector(), 96);
  EXPECT_EQ(pos.xOffset(), xOffset);
  EXPECT_EQ(pos.yOffset(), yOffset);
  EXPECT_EQ(pos.zOffset(), 345);
  const auto gameCoordinate = pos.toGameCoordinate();
  EXPECT_EQ(gameCoordinate.x, 6348);
  EXPECT_EQ(gameCoordinate.y, 802);
}