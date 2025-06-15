#ifndef CHARACTER_DETAIL_DIALOG_HPP_
#define CHARACTER_DETAIL_DIALOG_HPP_

#include "textureToQImage.hpp"

#include <silkroad_lib/scalar_types.hpp>
#include <silkroad_lib/pk2/gameData.hpp>

#include <QDialog>
#include <QElapsedTimer>
#include <QList>
#include <QProgressBar>
#include <QTimer>
#include <QString>

namespace Ui {
class CharacterDetailDialog;
}

struct SkillCooldown {
  sro::scalar_types::ReferenceSkillId skillId{0};
  int remainingMs{0};
};

struct CharacterData {
  int currentHp{0};
  int maxHp{0};
  int currentMp{0};
  int maxMp{0};
  QString stateMachine;
  QList<SkillCooldown> skillCooldowns;
};

class CharacterDetailDialog : public QDialog {
  Q_OBJECT
public:
  explicit CharacterDetailDialog(const sro::pk2::GameData &gameData, QWidget *parent = nullptr);
  ~CharacterDetailDialog();

  void setCharacterName(const QString &name);
  void updateCharacterData(const CharacterData &data);

public slots:
  void onCharacterDataUpdated(QString name, CharacterData data);

private:
  struct CooldownItem {
    sro::scalar_types::ReferenceSkillId skillId{0};
    int startMs{0};
    QElapsedTimer timer;
    QProgressBar *bar{nullptr};
    QString skillName;
  };

  Ui::CharacterDetailDialog *ui_;
  QString name_;
  const sro::pk2::GameData &gameData_;
  QTimer *cooldownTimer_{nullptr};
  QList<CooldownItem> cooldownItems_;

  void updateCooldownDisplays();
};

#include <QMetaType>

Q_DECLARE_METATYPE(SkillCooldown)
Q_DECLARE_METATYPE(CharacterData)

#endif // CHARACTER_DETAIL_DIALOG_HPP_
