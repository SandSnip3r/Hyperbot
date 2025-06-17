#ifndef CHARACTER_CELL_WIDGET_HPP_
#define CHARACTER_CELL_WIDGET_HPP_

#include "characterDetailDialog.hpp"
#include "barStyles.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QWidget>
#include <QString>
#include <QList>

class QLabel;
class QProgressBar;
class QListWidget;

class CharacterCellWidget : public QWidget {
  Q_OBJECT
public:
  explicit CharacterCellWidget(const sro::pk2::GameData &gameData,
                               QWidget *parent = nullptr);

  void setCharacterName(const QString &name);
  void updateCharacterData(const CharacterData &data);

protected:
  void mouseDoubleClickEvent(QMouseEvent *event) override;

private:
  void toggleExpanded();

  const sro::pk2::GameData &gameData_;
  QString name_;
  bool expanded_{false};

  QLabel *nameLabel_;
  QProgressBar *hpBar_;
  QProgressBar *mpBar_;
  QLabel *stateLabel_;
  QListWidget *skillCooldownList_;
  QWidget *detailsContainer_;
};

#endif // CHARACTER_CELL_WIDGET_HPP_
