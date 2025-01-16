#include "alchemyTableBuilder.hpp"
#include "probability.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <algorithm>
#include <functional>
#include <random>

using namespace std;

mt19937 createRandomEngine() {
  random_device rd;
  array<int, mt19937::state_size> seed_data;
  generate_n(seed_data.data(), seed_data.size(), ref(rd));
  seed_seq seq(begin(seed_data), end(seed_data));
  return mt19937(seq);
}

double expectedNumberOfElixirs(int goalPlus, const std::array<double,255> &probs) {
  if (goalPlus == 0) {
    return 0;
  }
  return (expectedNumberOfElixirs(goalPlus-1, probs) + 1) / probs[goalPlus-1];
}

void printExpectedValues(const std::array<double,255> &probs) {
  std::array<double,255> expectedValues;
  expectedValues[0] = 0.0;
  for (size_t i=1; i<expectedValues.size(); ++i) {
    expectedValues[i] = (expectedValues[i-1] + 1) / probs[i-1];
  }
  for (size_t i=0; i<20; ++i) {
    // How much does using a lucky stone here help?
    const double normalCostOfFailure = 1 * probs[i] + expectedValues[i] * (1-probs[i]);
    const double costOfFailureWithLuckStone = 1 * std::min(1.0, probs[i] + AlchemyTableBuilder::kLuckStoneBonus) + expectedValues[i] * (1-std::min(1.0, probs[i] + AlchemyTableBuilder::kLuckStoneBonus));
    cout << '+' << i << ": " << expectedValues[i] << endl;
    cout << "  Expected number of elixirs per attempt from here: " << normalCostOfFailure << ", but with a luck stone: " << costOfFailureWithLuckStone << ". Difference is " << normalCostOfFailure-costOfFailureWithLuckStone << endl;
  }
}

int main(){
  AlchemyTableBuilder builder;
  // builder.setUseLuckyPowder(true);
  // builder.setUsePremium(true);
  // builder.setAvatarBonus(0.04);
  // builder.useLuckStoneAtAndAbove(int level, bool use) {

  builder.build();
  for (const auto i : builder.get()) {
    cout << i << ',';
  }
  cout << endl;
  printExpectedValues(builder.get());

  // builder.useLuckStoneAt(4, true);
  // builder.build();
  // printExpectedValues(builder.get());

  // for (int i=0; i<20; ++i) {
  //   cout << builder.get()[i] << endl;
  // }
  return 0;
}