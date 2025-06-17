#ifndef CHARACTER_CARD_WIDGET_HPP_
#define CHARACTER_CARD_WIDGET_HPP_

#include <QFrame>
#include <QLabel>
#include <QProgressBar>
#include <QColor>

class CharacterCardWidget : public QFrame {
  Q_OBJECT
public:
  explicit CharacterCardWidget(const QString &name, QWidget *parent = nullptr);

  void setName(const QString &name);
  const QString &name() const { return name_; }
  void updateStatus(int currentHp, int maxHp, int currentMp, int maxMp);
  void setState(const QString &state);
  void setPairColor(const QColor &color);

signals:
  void clicked(const QString &name);

protected:
  void mousePressEvent(QMouseEvent *event) override;

private:
  QLabel *thumbnailLabel_;
  QProgressBar *hpBar_;
  QProgressBar *mpBar_;
  QLabel *stateLabel_;
  QString name_;
};

#endif // CHARACTER_CARD_WIDGET_HPP_
