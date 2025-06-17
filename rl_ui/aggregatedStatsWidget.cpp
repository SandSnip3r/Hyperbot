#include "aggregatedStatsWidget.hpp"

#include <QVBoxLayout>

AggregatedStatsWidget::AggregatedStatsWidget(QWidget *parent)
    : QWidget(parent),
      averageHpLabel_(new QLabel(tr("Average HP: 0%"), this)),
      activeDuelsLabel_(new QLabel(tr("Active Duels: 0"), this)) {
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->setContentsMargins(2, 2, 2, 2);
  layout->addWidget(averageHpLabel_);
  layout->addWidget(activeDuelsLabel_);
  setLayout(layout);
}

void AggregatedStatsWidget::onCharacterDataUpdated(QString name,
                                                   CharacterData data) {
  characterData_[name] = data;
  updateLabels();
}

void AggregatedStatsWidget::updateLabels() {
  if (characterData_.isEmpty()) {
    averageHpLabel_->setText(tr("Average HP: 0%"));
    activeDuelsLabel_->setText(tr("Active Duels: 0"));
    return;
  }
  double total = 0.0;
  int count = 0;
  for (const CharacterData &data : characterData_) {
    if (data.maxHp > 0) {
      total += static_cast<double>(data.currentHp) / data.maxHp;
      ++count;
    }
  }
  int avgPercent = count > 0 ? static_cast<int>((total / count) * 100.0) : 0;
  averageHpLabel_->setText(tr("Average HP: %1%" ).arg(avgPercent));
  activeDuelsLabel_->setText(tr("Active Duels: 0"));
}
