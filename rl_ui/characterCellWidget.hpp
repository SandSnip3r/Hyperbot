#ifndef CHARACTER_CELL_WIDGET_HPP_
#define CHARACTER_CELL_WIDGET_HPP_

#include "characterDetailDialog.hpp"
#include "barStyles.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QFrame>

class QLabel;
class QProgressBar;
class QVBoxLayout;

class CharacterCellWidget : public QFrame {
  Q_OBJECT
public:
  explicit CharacterCellWidget(const sro::pk2::GameData &gameData,
                               QWidget *parent = nullptr);
  void setCharacterName(const QString &name);
  QString characterName() const;
  void updateCharacterData(const CharacterData &data);

protected:
  void mousePressEvent(QMouseEvent *event) override;

private:
  QString name_;
  bool expanded_{false};
  const sro::pk2::GameData &gameData_;
  QLabel *nameLabel_{nullptr};
  QProgressBar *hpBar_{nullptr};
  QProgressBar *mpBar_{nullptr};
  QLabel *stateLabel_{nullptr};
  QWidget *detailsWidget_{nullptr};
  QLabel *stateMachineLabel_{nullptr};

  void setExpanded(bool expanded);
};

#endif // CHARACTER_CELL_WIDGET_HPP_
