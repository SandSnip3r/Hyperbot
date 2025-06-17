#include "aggregatedStatsWidget.hpp"

#include <QVBoxLayout>

AggregatedStatsWidget::AggregatedStatsWidget(QWidget *parent)
    : QWidget(parent), label_(new QLabel(tr("Average HP: 0%"), this)) {
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->addWidget(label_);
  setLayout(layout);
}

void AggregatedStatsWidget::onAggregatedStats(float avgHpPercent) {
  label_->setText(tr("Average HP: %1%" ).arg(QString::number(avgHpPercent, 'f', 1)));
}
