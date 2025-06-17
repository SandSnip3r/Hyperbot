#ifndef DASHBOARD_WIDGET_HPP_
#define DASHBOARD_WIDGET_HPP_

#include "characterDetailDialog.hpp"
#include "characterTileWidget.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QStringList>
#include <QWidget>
#include <QMap>
#include <QGridLayout>
#include <QScrollArea>

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
  void onHyperbotConnected();

signals:
  void characterDataUpdated(QString name, CharacterData data);

private:
  Ui::DashboardWidget *ui;
  QMap<QString, CharacterData> characterData_;
  QMap<QString, CharacterTileWidget *> tiles_;
  const sro::pk2::GameData &gameData_;
  QWidget *gridContainer_{nullptr};
  QGridLayout *gridLayout_{nullptr};
  int columns_{4};

  CharacterTileWidget *ensureTileForCharacter(const QString &name);
  void updateGridPositions();
};

#endif // DASHBOARD_WIDGET_HPP_
