#ifndef CHARACTER_DETAIL_WIDGET_HPP_
#define CHARACTER_DETAIL_WIDGET_HPP_

#include "textureToQImage.hpp"

#include <silkroad_lib/scalar_types.hpp>
#include <silkroad_lib/pk2/gameData.hpp>

#include <QWidget>
#include <QList>
#include <QProgressBar>
#include <QListWidgetItem>
#include <QTimer>
#include <QHash>
#include <QPixmap>
#include <QString>

namespace Ui {
class CharacterDetailWidget;
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

class CharacterDetailWidget : public QWidget {
  Q_OBJECT
public:
  explicit CharacterDetailWidget(const sro::pk2::GameData &gameData,
                                 QWidget *parent = nullptr);
  ~CharacterDetailWidget();

  void setCharacterName(const QString &name);
  void updateCharacterData(const CharacterData &data);

public slots:
  void onCharacterDataUpdated(QString name, CharacterData data);

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

  Ui::CharacterDetailWidget *ui_;
  QString name_;
  const sro::pk2::GameData &gameData_;
  static QTimer *sharedCooldownTimer_;
  static int activeWidgetCount_;
  QHash<sro::scalar_types::ReferenceSkillId, CooldownItem> cooldownItems_;
  QHash<sro::scalar_types::ReferenceSkillId, QPixmap> iconCache_;

  QPixmap getIconForSkillId(sro::scalar_types::ReferenceSkillId skillId);
  void updateCooldownDisplays();
};

#include <QMetaType>

Q_DECLARE_METATYPE(SkillCooldown)
Q_DECLARE_METATYPE(CharacterData)

#endif // CHARACTER_DETAIL_WIDGET_HPP_
