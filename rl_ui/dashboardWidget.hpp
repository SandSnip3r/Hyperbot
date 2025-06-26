#ifndef DASHBOARD_WIDGET_HPP_
#define DASHBOARD_WIDGET_HPP_

#include "characterDetailDialog.hpp"
#include <silkroad_lib/pk2/gameData.hpp>
#include <silkroad_lib/scalar_types.hpp>

#include <QStringList>
#include <QWidget>
#include <QMap>
#include <QVector>

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
  void onQValues(QString name, QVector<float> qValues);
  void onItemCount(QString name, sro::scalar_types::ReferenceObjectId itemRefId,
                   int count);
  void clearStatusTable();
  void onHyperbotConnected();

signals:
  void characterStatusUpdated(QString name, int currentHp, int maxHp,
                              int currentMp, int maxMp);
  void activeStateMachineUpdated(QString name, QString stateMachine);
  void skillCooldownsUpdated(QString name, QList<SkillCooldown> cooldowns);
  void qValuesUpdated(QString name, QVector<float> qValues);
  void itemCountUpdated(QString name,
                        sro::scalar_types::ReferenceObjectId itemRefId,
                        int count);

private:
  Ui::DashboardWidget *ui;
  QMap<QString, CharacterData> characterData_;
  QMap<QString, CharacterDetailDialog *> detailDialogs_;
  const sro::pk2::GameData &gameData_;
  int ensureRowForCharacter(const QString &name);
  void showCharacterDetail(int row, int column);
};

#endif // DASHBOARD_WIDGET_HPP_
