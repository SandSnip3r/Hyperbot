#ifndef DASHBOARD_WIDGET_HPP_
#define DASHBOARD_WIDGET_HPP_

#include "characterDetailDialog.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QStringList>
#include <QWidget>
#include <QMap>
#include <QTreeWidgetItem>
#include <QHash>

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
  struct CharacterWidgets {
    QTreeWidgetItem *item{nullptr};
    QProgressBar *hpBar{nullptr};
    QProgressBar *mpBar{nullptr};
  };

  struct PairWidgets {
    QTreeWidgetItem *item{nullptr};
    QProgressBar *hpBar{nullptr};
    QProgressBar *mpBar{nullptr};
    QString first;
    QString second;
  };

  QMap<QString, CharacterData> characterData_;
  QHash<QString, CharacterWidgets> characterWidgets_;
  QHash<int, PairWidgets> pairWidgets_;
  QMap<QString, CharacterDetailDialog *> detailDialogs_;
  const sro::pk2::GameData &gameData_;
  CharacterWidgets &ensureCharacterWidgets(const QString &name);
  PairWidgets &ensurePairWidgets(int pairId);
  static int characterId(const QString &name);
  static int pairIdForName(const QString &name);
  void updatePairSummary(int pairId);
  void showCharacterDetail(QTreeWidgetItem *item, int column);
};

#endif // DASHBOARD_WIDGET_HPP_
