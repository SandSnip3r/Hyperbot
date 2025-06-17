#ifndef CARD_DASHBOARD_WIDGET_HPP_
#define CARD_DASHBOARD_WIDGET_HPP_

#include "characterCardWidget.hpp"
#include "characterDetailDialog.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QLineEdit>
#include <QScrollArea>
#include <QWidget>
#include <QGridLayout>
#include <QMap>

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
  void clearCards();
  void onHyperbotConnected();

signals:
  void cardSelected(QString name);
  void characterDataUpdated(QString name, CharacterData data);

private slots:
  void filterTextChanged(const QString &text);

private:
  QLineEdit *filterEdit_;
  QScrollArea *scrollArea_;
  QWidget *gridWidget_;
  QGridLayout *gridLayout_;
  QMap<QString, CharacterCardWidget *> cardWidgets_;
  QMap<QString, CharacterData> characterData_;
  const sro::pk2::GameData &gameData_;

  CharacterCardWidget *ensureCard(const QString &name);
  QColor colorForPair(const QString &name) const;
  void applyFilter();
};

#endif // CARD_DASHBOARD_WIDGET_HPP_
