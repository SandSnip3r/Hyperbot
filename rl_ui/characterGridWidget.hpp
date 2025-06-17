#ifndef CHARACTER_GRID_WIDGET_HPP_
#define CHARACTER_GRID_WIDGET_HPP_

#include "characterCellWidget.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QWidget>
#include <QGridLayout>
#include <QMap>

class CharacterGridWidget : public QWidget {
  Q_OBJECT
public:
  explicit CharacterGridWidget(const sro::pk2::GameData &gameData,
                               QWidget *parent = nullptr);

public slots:
  void onCharacterStatusReceived(QString name, int currentHp, int maxHp,
                                 int currentMp, int maxMp);
  void onActiveStateMachine(QString name, QString stateMachine);
  void onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns);
  void clearCharacters();

signals:
  void characterDataUpdated(QString name, CharacterData data);

private:
  int ensureCellForCharacter(const QString &name);

  const sro::pk2::GameData &gameData_;
  QGridLayout *gridLayout_;
  QMap<QString, CharacterCellWidget *> cells_;
  QMap<QString, CharacterData> characterData_;
  int columns_{4};
};

#endif // CHARACTER_GRID_WIDGET_HPP_
