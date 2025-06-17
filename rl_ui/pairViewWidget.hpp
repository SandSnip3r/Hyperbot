#ifndef PAIR_VIEW_WIDGET_HPP_
#define PAIR_VIEW_WIDGET_HPP_

#include "characterDetailDialog.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QWidget>
#include <QMap>
#include <QVector>

namespace Ui {
class PairViewWidget;
}

class PairViewWidget : public QWidget {
  Q_OBJECT
public:
  explicit PairViewWidget(const sro::pk2::GameData &gameData,
                          QWidget *parent = nullptr);
  ~PairViewWidget();

public slots:
  void onCharacterStatusReceived(QString name, int currentHp, int maxHp,
                                 int currentMp, int maxMp);
  void onActiveStateMachine(QString name, QString stateMachine);
  void onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns);
  void onHyperbotConnected();
  void clearData();

private slots:
  void onPairSelectionChanged(int currentRow);

private:
  void ensureRowCount(int row);
  void updateDetailViews();

  Ui::PairViewWidget *ui_;
  const sro::pk2::GameData &gameData_;
  QVector<QString> teamA_;
  QVector<QString> teamB_;
  QMap<QString, CharacterData> characterData_;
  int selectedRow_{-1};
};

#endif // PAIR_VIEW_WIDGET_HPP_
