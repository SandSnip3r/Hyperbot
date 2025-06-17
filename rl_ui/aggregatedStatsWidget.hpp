#ifndef AGGREGATED_STATS_WIDGET_HPP_
#define AGGREGATED_STATS_WIDGET_HPP_

#include "characterDetailDialog.hpp"

#include <QLabel>
#include <QMap>
#include <QWidget>

class AggregatedStatsWidget : public QWidget {
  Q_OBJECT
public:
  explicit AggregatedStatsWidget(QWidget *parent = nullptr);

public slots:
  void onCharacterDataUpdated(QString name, CharacterData data);

private:
  QLabel *averageHpLabel_;
  QLabel *activeDuelsLabel_;
  QMap<QString, CharacterData> characterData_;

  void updateLabels();
};

#endif // AGGREGATED_STATS_WIDGET_HPP_
