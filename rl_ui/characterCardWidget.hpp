#ifndef CHARACTER_CARD_WIDGET_HPP_
#define CHARACTER_CARD_WIDGET_HPP_

#include "characterDetailDialog.hpp"

#include <QFrame>
#include <QLabel>
#include <QProgressBar>

class CharacterCardWidget : public QFrame {
  Q_OBJECT
public:
  explicit CharacterCardWidget(QWidget *parent = nullptr);

  void setCharacterName(const QString &name);
  void updateCharacterData(const CharacterData &data);

signals:
  void cardClicked(QString name);

protected:
  void mousePressEvent(QMouseEvent *event) override;

private:
  QLabel *thumbnailLabel_;
  QLabel *nameLabel_;
  QLabel *stateLabel_;
  QProgressBar *hpBar_;
  QProgressBar *mpBar_;
  QString name_;
};

#endif // CHARACTER_CARD_WIDGET_HPP_
