#ifndef CHARACTER_TILE_WIDGET_HPP_
#define CHARACTER_TILE_WIDGET_HPP_

#include "characterDetailDialog.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QFrame>
#include <QLabel>
#include <QProgressBar>
#include <QListWidget>
#include <QVBoxLayout>

class CharacterTileWidget : public QFrame {
  Q_OBJECT
public:
  explicit CharacterTileWidget(const sro::pk2::GameData &gameData,
                               QWidget *parent = nullptr);

  void setCharacterName(const QString &name);
  void updateCharacterData(const CharacterData &data);

signals:
  void expandedChanged(bool expanded);

protected:
  void mousePressEvent(QMouseEvent *event) override;

private:
  bool expanded_{false};
  QVBoxLayout *layout_{nullptr};
  QLabel *nameLabel_{nullptr};
  QProgressBar *hpBar_{nullptr};
  QProgressBar *mpBar_{nullptr};
  QListWidget *cooldownList_{nullptr};
  QLabel *stateLabel_{nullptr};
  const sro::pk2::GameData &gameData_;

  void updateDetailVisibility();
};

#endif // CHARACTER_TILE_WIDGET_HPP_
