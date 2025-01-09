#ifndef PROBABILITY_HPP_
#define PROBABILITY_HPP_

#include "cache.h"

#include <array>
#include <string>
#include <iostream>
#include <unordered_map>

class Probability {
public:
  Probability(const std::array<double,255> &probabilityTable, double probabilityThreshold = 1.0) : probabilityTable_(probabilityTable), probabilityThreshold_(probabilityThreshold) {}

  double calculate(int current, int countAvailable, int target, int recursionIndent=0) const {
    const auto indent = std::string(recursionIndent, ' ');
    std::cout << indent << "Want to achieve +" << target << ". Have " << countAvailable << " elixirs. Currently at +" << current << '.' << std::endl;
    if (current == target) {
      std::cout << indent << "Met goal. 1.0" << std::endl;
      return 1.0;
    }
    if (current + countAvailable < target) {
      std::cout << indent << "Not enough elixirs. 0.0" << std::endl;
      return 0.0;
    }
    const auto argumentTuple = std::make_tuple(current,countAvailable,target);
    if (const auto it = cache_.find(argumentTuple); it != cache_.end()) {
      std::cout << indent << "cached value of " << current << ',' << countAvailable << ',' << target << " is " << it->second << std::endl;
      return it->second;
    }
    const double prob = probabilityOf(current);
    const auto a = prob;
    const auto b = (1-prob)*calculate(0, countAvailable-1, current, recursionIndent+1);
    std::cout << indent << "Prob of up one " << a*100 << "% and prob of back to where we are " << b*100 << '%' << std::endl;
    const bool shouldTryToGoUp1 = (a + b) >= probabilityThreshold_;
    double result = 0.0;
    if (shouldTryToGoUp1) {
      std::cout << indent << "Want to try" << std::endl;
      result = prob  * calculate(current+1, countAvailable-1, target, recursionIndent+1) + 
            (1-prob) * calculate(0, countAvailable-1, target, recursionIndent+1);
      std::cout << indent << "Result is " << result << std::endl;
    } else {
      std::cout << indent << "Don't want to try" << std::endl;
    }
    cache_[argumentTuple] = result;
    return result;
  }

  double probabilityOf(int plus) const {
    return probabilityTable_[plus];
  }

  double getRatio() const {
    return probabilityThreshold_;
  }
private:
  // probabilityTable_[x] = the probability of a success when going from x to x+1, i.e. probabilityTable_[0] is the probability of successfully going to +1
  const std::array<double,255> probabilityTable_;
  const double probabilityThreshold_;
  mutable std::unordered_map<std::tuple<int, int, int>, double> cache_;
};

#endif // PROBABILITY_HPP_