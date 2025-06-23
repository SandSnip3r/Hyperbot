#ifndef CHARACTER_DETAIL_DIALOG_HPP_
#define CHARACTER_DETAIL_DIALOG_HPP_

#include "textureToQImage.hpp"

#include <silkroad_lib/scalar_types.hpp>
#include <silkroad_lib/pk2/gameData.hpp>

#include <QDialog>
#include <QList>
#include <QProgressBar>
#include <QListWidgetItem>
#include <QWidget>
#include <QTimer>
#include <QHash>
#include <QPixmap>
#include <QString>
#include <QVector>

#include "hyperbot.hpp"

namespace Ui {
class CharacterDetailDialog;
}

struct SkillCooldown {
  sro::scalar_types::ReferenceSkillId skillId{0};
  int remainingMs{0};
  qint64 timestampMs{0};
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
  explicit CharacterDetailDialog(const sro::pk2::GameData &gameData,
                                 Hyperbot &hyperbot,
                                 QWidget *parent = nullptr);
  ~CharacterDetailDialog();

  void setCharacterName(const QString &name);
  void updateCharacterData(const CharacterData &data);

public slots:
  void onCharacterDataUpdated(QString name, CharacterData data);
  void onQValuesReceived(QString name, QVector<float> qValues);

private:
  struct CooldownItem {
    sro::scalar_types::ReferenceSkillId skillId{0};
    int totalMs{0};
    int remainingMs{0};
    qint64 timestampMs{0};
    QListWidgetItem *item{nullptr};
    QWidget *container{nullptr};
    QProgressBar *bar{nullptr};
    QString skillName;
  };

  class CooldownListItem : public QListWidgetItem {
  public:
    using QListWidgetItem::QListWidgetItem;

    bool operator<(const QListWidgetItem &other) const override {
      return data(Qt::UserRole).toInt() < other.data(Qt::UserRole).toInt();
    }
  };

  Ui::CharacterDetailDialog *ui_;
  QString name_;
  const sro::pk2::GameData &gameData_;
  Hyperbot &hyperbot_;
  static QTimer *sharedCooldownTimer_;
  static int activeDialogCount_;
  QHash<sro::scalar_types::ReferenceSkillId, CooldownItem> cooldownItems_;
  QHash<sro::scalar_types::ReferenceSkillId, QPixmap> iconCache_;
  QVector<QProgressBar*> qValueBars_;

  QPixmap getIconForSkillId(sro::scalar_types::ReferenceSkillId skillId);
  void updateCooldownDisplays();
};

#include <QMetaType>

Q_DECLARE_METATYPE(SkillCooldown)
Q_DECLARE_METATYPE(CharacterData)

#endif // CHARACTER_DETAIL_DIALOG_HPP_
