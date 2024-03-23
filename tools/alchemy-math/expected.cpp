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

void printExpectedValues(const std::array<double,255> &probs) {
  auto ps = [&](int n) {
    // Probability of n successes.
    double res=1.0;
    for (int i=0; i<n; ++i) {
      res *= probs[i];
    }
    return res;
  };

  auto p = [&](int n) {
    // Probability of n-1 successes then one failure.
    return ps(n-1) * (1-probs[n-1]);
  };

  auto expected = [&](int n) {
    double numerator = 0.0;
    for (int i=1; i<=n; ++i) {
      numerator += i*p(i);
    }
    numerator += n * ps(n);

    const double denominator = [&](){
      double res = 1-probs[n-1]; // Fail n-1
      for (int i=n-2; i>=0; --i) {
        res *= probs[i];
        res += 1.0-probs[i];
      }
      return 1.0-res;
    }();

    cout << numerator << '/' << denominator << endl;
    return numerator / denominator;
  };

  auto simulated = [&](int n) {
    static auto eng = createRandomEngine();
    vector<bernoulli_distribution> dists;
    for (int i=0; i<=n; ++i) {
      dists.emplace_back(probs[i]);
    }
    constexpr const int kTrialCount = 100'000;
    uint64_t count=0;
    for (int i=0; i<kTrialCount; ++i) {
      int current=0;
      while (current < n) {
        if (dists[current](eng)) {
          ++current;
        } else {
          current = 0;
        }
        ++count;
      }//
    }
    return count / static_cast<double>(kTrialCount);
  };

  for (int i=0; i<=250; ++i) {
    const auto e = expected(i);
    // const auto s = simulated(i);
    // cout << i << ": " << e << ", " << s << endl;
    cout << i << ": " << e << endl;
  }
}

int main(){
  AlchemyTableBuilder builder;
  builder.useLuckyPowder(true);

  builder.build();
  printExpectedValues(builder.get());

  // builder.useLuckStoneAt(4, true);
  // builder.build();
  // printExpectedValues(builder.get());

  // for (int i=0; i<20; ++i) {
  //   cout << builder.get()[i] << endl;
  // }
  return 0;
}