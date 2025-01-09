#include "alchemyTableBuilder.hpp"
#include "probability.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <array>
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

int main() {
  AlchemyTableBuilder alchemyTableBuilder;
  alchemyTableBuilder.useLuckyPowder(true);
  // alchemyTableBuilder.useLuckStone(true);
  // alchemyTableBuilder.usePremium(true);
  // alchemyTableBuilder.useAvatar(true);
  alchemyTableBuilder.build();

  // const double kRatio = 2.0;
  // const double kProbFromRatio = kRatio/(kRatio+1);
  const double kProbFromRatio = 0.8;
  Probability prob(alchemyTableBuilder.get(), kProbFromRatio);

  // ================================================================================
  // ================================================================================
  // ================================================================================

  {
    const int current = 1;
    const int numElixirs = 2;
    const int goal = 2;
    cout << "Conducting experiment. Current: +" << current << ", num elixirs: " << numElixirs << ", goal: +" << goal << ". Threshold: " << kProbFromRatio*100 << '%' << endl;
    double res = prob.calculate(current, numElixirs, goal);
    cout << "Test result: " << res << endl;
    return 0;
  }

  // ================================================================================
  // ================================================================================
  // ================================================================================

  const int kTrialStart=0;
  const int kTrialGoal=3;
  const int kTrialNumberOfElixirs = 100;
  // constexpr const int kTrialCount = 1;
  // constexpr const int kTrialCount = 10'000;
  constexpr const int kTrialCount = 10'000'000;
  
  vector<bernoulli_distribution> distributions;
  for (int i=0; i<=kTrialGoal; ++i) {
    distributions.emplace_back(prob.probabilityOf(i));
  }
  auto eng = createRandomEngine();

  int successCount = 0;
  for (int i=0; i<kTrialCount; ++i) {
    int current = kTrialStart;
    int numberOfElixirs = kTrialNumberOfElixirs;
    while (current < kTrialGoal && numberOfElixirs > 0) {
      const auto probabilityOfGoingUp1 = prob.probabilityOf(current);
      const bool shouldTryToGoUp1 = probabilityOfGoingUp1 + (1-probabilityOfGoingUp1)*prob.calculate(0, numberOfElixirs-1, current) >= kProbFromRatio;
      if (shouldTryToGoUp1) {
        // Try
        bool result = distributions[current](eng);
        // cout << "Tried " << current+1;
        if (result) {
          // cout << ". Succeeded!" << endl;
          ++current;
        } else {
          // cout << ". Failed!" << endl;
          current = 0;
        }
        --numberOfElixirs;
      } else {
        // Not enough, quit here.
        break;
      }
    }
    if (current == kTrialGoal) {
      ++successCount;
    }
    // cout << "Ending at " << current << " with " << numberOfElixirs << " elixirs left" << endl;
  }
  cout << "Final ratio: " << successCount / static_cast<double>(kTrialCount) << endl;
  cout << "Function says " << prob.calculate(kTrialStart, kTrialNumberOfElixirs, kTrialGoal) << endl;
}