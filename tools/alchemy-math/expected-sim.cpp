#include <iostream>
#include <vector>
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

double expected(int inARow, int goal, const vector<double> &probabilities) {
  if (inARow == 0) {
    return 0;
  }
  cout << "Looking for " << inARow << ',' << goal << endl;
  double probOfSuccess = 1;
  for (int i=goal-inARow; i>=0; --i) {
    probOfSuccess *= probabilities[i];
  }
  cout << "Probability is " << probOfSuccess << endl;
  return (probOfSuccess*(expected(inARow-1, goal, probabilities) + 1) + 1 - probOfSuccess) / probOfSuccess;
}

double simulate(int target, const vector<double> &probabilities) {
  static auto eng = createRandomEngine();
  vector<bernoulli_distribution> dists;
  for (double prob : probabilities) {
    dists.emplace_back(prob);
  }
  constexpr const int kTrialCount = 1'000'000;
  uint64_t count = 0;
  for (int i=0; i<kTrialCount; ++i) {
    int current = 0;
    while (current < target) {
      if (dists[current](eng)) {
        ++current;
      } else {
        current = 0;
      }
      cout << current << endl;
      ++count;
    }
  }
  return count / static_cast<double>(kTrialCount);
}

int main() {
  constexpr const int kMaxPlus = 30;
  std::vector<double> probabilities = {1.0,0.7,0.5,0.27,0.25,0.25,0.25,0.25,0.25};
  for (int i=probabilities.size(); i<kMaxPlus; ++i) {
    probabilities.push_back(0.2);
  }
  // for (int i=0; i<5; ++i) {
  //   const auto res = expected(i, i, probabilities);
  //   cout << "The expected number of elixirs to get +" << i << " from 0 is " << res << endl;
  //   const auto simRes = simulate(i, probabilities);
  //   cout << "The simulated number of elixirs to get +" << i << " from 0 is " << simRes << endl;
  // }
  // const auto res = expected(3, 3, probabilities);
  simulate(2, probabilities);
}