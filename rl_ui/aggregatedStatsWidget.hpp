#ifndef AGGREGATED_STATS_WIDGET_HPP_
#define AGGREGATED_STATS_WIDGET_HPP_

#include <QWidget>
#include <QLabel>

class AggregatedStatsWidget : public QWidget {
  Q_OBJECT
public:
  explicit AggregatedStatsWidget(QWidget *parent = nullptr);

public slots:
  void onAggregatedStats(float avgHpPercent);

private:
  QLabel *label_;
};

#endif // AGGREGATED_STATS_WIDGET_HPP_
