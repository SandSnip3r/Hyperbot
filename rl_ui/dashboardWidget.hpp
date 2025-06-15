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

public slots:
  void onCharacterStatusReceived(QString name, int currentHp, int maxHp,
                                 int currentMp, int maxMp);
  void onActiveStateMachine(QString name, QString stateMachine);
  void onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns);
  void clearStatusTable();

signals:
  void characterDataUpdated(QString name, CharacterData data);

private:
  Ui::DashboardWidget *ui;
  QMap<QString, CharacterData> characterData_;
  const sro::pk2::GameData &gameData_;
  int ensureRowForCharacter(const QString &name);
  void showCharacterDetail(int row, int column);
};

#endif // DASHBOARD_WIDGET_HPP_
