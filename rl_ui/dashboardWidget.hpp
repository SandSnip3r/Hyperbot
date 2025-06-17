#ifndef DASHBOARD_WIDGET_HPP_
#define DASHBOARD_WIDGET_HPP_

#include "characterCardWidget.hpp"
#include "characterDetailDialog.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QStringList>
#include <QWidget>
#include <QMap>
#include <QGridLayout>
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
  void clearStatusTable();
  void onHyperbotConnected();

signals:
  void characterDataUpdated(QString name, CharacterData data);

private:
  Ui::DashboardWidget *ui;
  QMap<QString, CharacterData> characterData_;
  QMap<QString, CharacterDetailDialog *> detailDialogs_;
  QMap<QString, CharacterCardWidget *> cardWidgets_;
  QWidget *gridContainer_;
  QGridLayout *gridLayout_;
  const sro::pk2::GameData &gameData_;

  CharacterCardWidget *ensureCardForCharacter(const QString &name);
  void showCharacterDetail(QString name);
  QColor colorForPair(int pair) const;
};

#endif // DASHBOARD_WIDGET_HPP_
