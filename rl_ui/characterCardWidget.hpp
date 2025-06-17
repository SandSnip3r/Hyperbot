#ifndef CHARACTER_CARD_WIDGET_HPP_
#define CHARACTER_CARD_WIDGET_HPP_

#include "barStyles.hpp"

#include <QFrame>
#include <QImage>

class QLabel;
class QProgressBar;

namespace Ui {
class CharacterCardWidget;
}

class CharacterCardWidget : public QFrame {
  Q_OBJECT
public:
  explicit CharacterCardWidget(QWidget *parent = nullptr);
  ~CharacterCardWidget();

  void setCharacterName(const QString &name);
  void setThumbnail(const QImage &image);
  void updateHpMp(int hp, int maxHp, int mp, int maxMp);
  void setStateText(const QString &state);

signals:
  void clicked();

protected:
  void mousePressEvent(QMouseEvent *event) override;

private:
  Ui::CharacterCardWidget *ui_;
  QLabel *thumbnailLabel_{nullptr};
  QLabel *nameLabel_{nullptr};
  QLabel *stateLabel_{nullptr};
  QProgressBar *hpBar_{nullptr};
  QProgressBar *mpBar_{nullptr};
};

#endif // CHARACTER_CARD_WIDGET_HPP_
