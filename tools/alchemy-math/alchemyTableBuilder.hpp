#ifndef ALCHEMYTABLEBUILDER_HPP_
#define ALCHEMYTABLEBUILDER_HPP_

#include <algorithm>
#include <array>

class AlchemyTableBuilder {
public:
  AlchemyTableBuilder() {
    // Fill base rates.
    int index=0;
    baseRates_[index++] = 0.50;
    baseRates_[index++] = 0.40;
    baseRates_[index++] = 0.30;
    baseRates_[index++] = 0.19;
    baseRates_[index++] = 0.17;
    baseRates_[index++] = 0.17;
    baseRates_[index++] = 0.17;
    baseRates_[index++] = 0.17;
    baseRates_[index++] = 0.17;
    std::fill(baseRates_.begin()+index, baseRates_.end(), 0.12);

    // Fill lucky powder rates.
    index=0;
    powderBonuses_[index++] = 0.50;
    powderBonuses_[index++] = 0.30;
    powderBonuses_[index++] = 0.20;
    std::fill(powderBonuses_.begin()+index, powderBonuses_.end(), 0.08);
  }

  void useLuckyPowder(bool use) {
    useLuckyPowder_ = use;
  }

  void resetLuckStoneUse() {
    std::fill(useLuckStoneTable_.begin(), useLuckStoneTable_.end(), false);
  }

  void useLuckStoneAt(int level, bool use) {
    useLuckStoneTable_[level] = true;
  }

  void usePremium(bool use) {
    usePremium_ = use;
  }

  void setAvatarBonus(double bonus) {
    avatarBonus_ = bonus;
  }

  void build() {
    for (size_t i=0; i<table_.size(); ++i) {
      table_[i] = baseRates_[i];
      if (useLuckyPowder_) {
        table_[i] += powderBonuses_[i];
      }
      if (useLuckStoneTable_[i]) {
        table_[i] += kLuckStoneBonus;
      }
      if (usePremium_) {
        table_[i] += kPremiumStoneBonus;
      }
      table_[i] += avatarBonus_;
      table_[i] = std::clamp(table_[i], 0.0, 1.0);
    }
  }
  
  const std::array<double,255>& get() const {
    return table_;
  }
private:
  static constexpr const double kLuckStoneBonus{0.05};
  static constexpr const double kPremiumStoneBonus{0.05};
  std::array<double,255> baseRates_;
  std::array<double,255> powderBonuses_;
  std::array<bool,255> useLuckStoneTable_ = {};
  bool useLuckyPowder_{false};
  bool usePremium_{false};
  double avatarBonus_{0.0};
  std::array<double,255> table_;
};

#endif // ALCHEMYTABLEBUILDER_HPP_