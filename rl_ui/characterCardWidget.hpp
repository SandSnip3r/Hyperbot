#ifndef CHARACTER_CARD_WIDGET_HPP_
#define CHARACTER_CARD_WIDGET_HPP_

#include "characterDetailDialog.hpp"
#include "barStyles.hpp"

#include <QFrame>
#include <QLabel>
#include <QProgressBar>
#include <QPixmap>

class CharacterCardWidget : public QFrame {
  Q_OBJECT
public:
  explicit CharacterCardWidget(QWidget *parent = nullptr);

  void setCharacterName(const QString &name);
  void setPairColor(const QColor &color);
  void updateCharacterData(const CharacterData &data);
  void setThumbnail(const QPixmap &pixmap);

signals:
  void clicked(QString name);

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
