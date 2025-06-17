#ifndef DASHBOARD_WIDGET_HPP_
#define DASHBOARD_WIDGET_HPP_

#include "characterDetailDialog.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QStringList>
#include <QWidget>
#include <QMap>

namespace Ui {
class DashboardWidget;
}

class DashboardWidget : public QWidget {
  Q_OBJECT
public:
  explicit DashboardWidget(const sro::pk2::GameData &gameData,
                           QWidget *parent = nullptr);
  ~DashboardWidget();

  CharacterData getCharacterData(const QString &name) const;

public slots:
  void onCharacterStatusReceived(QString name, int currentHp, int maxHp,
                                 int currentMp, int maxMp);
  void onActiveStateMachine(QString name, QString stateMachine);
  void onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns);
  void clearStatusTable();
  void onHyperbotConnected();

signals:
  void characterDataUpdated(QString name, CharacterData data);
  void aggregatedStatsUpdated(float averageHpPercent);

 private:
  Ui::DashboardWidget *ui;
  QMap<QString, CharacterData> characterData_;
  QMap<QString, CharacterDetailDialog *> detailDialogs_;
  const sro::pk2::GameData &gameData_;
  QPoint dragStartPos_;
  int ensureRowForCharacter(const QString &name);
  void showCharacterDetail(int row, int column);
  void updateAggregatedStats();

  bool eventFilter(QObject *obj, QEvent *event) override;
};

#endif // DASHBOARD_WIDGET_HPP_
