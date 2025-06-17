#ifndef CARD_DASHBOARD_WIDGET_HPP_
#define CARD_DASHBOARD_WIDGET_HPP_

#include "characterCardWidget.hpp"
#include "characterDetailDialog.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QGridLayout>
#include <QMap>
#include <QScrollArea>
#include <QDockWidget>
#include <QWidget>

class CardDashboardWidget : public QWidget {
  Q_OBJECT
public:
  explicit CardDashboardWidget(const sro::pk2::GameData &gameData,
                               QWidget *parent = nullptr);
  ~CardDashboardWidget();

public slots:
  void onCharacterStatusReceived(QString name, int currentHp, int maxHp,
                                 int currentMp, int maxMp);
  void onActiveStateMachine(QString name, QString stateMachine);
  void onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns);
  void clearStatus();
  void onHyperbotConnected();

signals:
  void characterDataUpdated(QString name, CharacterData data);

private:
  QScrollArea *scrollArea_;
  QWidget *container_;
  QGridLayout *gridLayout_;
  const sro::pk2::GameData &gameData_;
  QMap<QString, CharacterCardWidget *> cardMap_;
  QMap<QString, CharacterData> characterData_;
  QMap<QString, QDockWidget *> dockMap_;

  CharacterCardWidget *ensureCard(const QString &name);
  void showCharacterDetail(QString name);
};

#endif // CARD_DASHBOARD_WIDGET_HPP_
