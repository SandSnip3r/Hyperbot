#ifndef DASHBOARD_WIDGET_HPP_
#define DASHBOARD_WIDGET_HPP_

#include "characterCellWidget.hpp"
#include "characterDetailDialog.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QStringList>
#include <QWidget>
#include <QMap>
#include <QGridLayout>

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
  void clearStatus();
  void onHyperbotConnected();
  void onCellExpandRequested(CharacterCellWidget *cell);

signals:
  void characterDataUpdated(QString name, CharacterData data);

private:
  Ui::DashboardWidget *ui;
  QGridLayout *gridLayout_;
  QMap<QString, CharacterCellWidget *> cellWidgets_;
  QMap<QString, CharacterData> characterData_;
  const sro::pk2::GameData &gameData_;
  int ensureCellForCharacter(const QString &name);
};

#endif // DASHBOARD_WIDGET_HPP_
