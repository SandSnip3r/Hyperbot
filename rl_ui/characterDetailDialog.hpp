#ifndef CHARACTER_DETAIL_DIALOG_HPP_
#define CHARACTER_DETAIL_DIALOG_HPP_

#include "textureToQImage.hpp"

#include <silkroad_lib/scalar_types.hpp>
#include <silkroad_lib/pk2/gameData.hpp>

#include <QDialog>
#include <QList>
#include <QString>
#include <QMap>
#include <QProgressBar>
#include <QTimer>
#include <QListWidgetItem>

#include <filesystem>
#include <memory>
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
  Ui::CharacterDetailDialog *ui_;
  QString name_;
  const sro::pk2::GameData &gameData_;
  struct CooldownWidget {
    QListWidgetItem *item{nullptr};
    QProgressBar *bar{nullptr};
    double totalMs{0};
    double remainingMs{0};
  };
  QMap<sro::scalar_types::ReferenceSkillId, CooldownWidget> cooldownWidgets_;
  QTimer updateTimer_;
  void updateCooldownBars();
};

#include <QMetaType>

Q_DECLARE_METATYPE(SkillCooldown)
Q_DECLARE_METATYPE(CharacterData)

#endif // CHARACTER_DETAIL_DIALOG_HPP_
