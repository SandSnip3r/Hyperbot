#include "characterCardWidget.hpp"
#include "ui_characterCardWidget.h"

#include <QLabel>
#include <QMouseEvent>
#include <QProgressBar>

CharacterCardWidget::CharacterCardWidget(QWidget *parent)
    : QFrame(parent), ui_(new Ui::CharacterCardWidget) {
  ui_->setupUi(this);
  thumbnailLabel_ = ui_->thumbnailLabel;
  nameLabel_ = ui_->nameLabel;
  stateLabel_ = ui_->stateLabel;
  hpBar_ = ui_->hpBar;
  mpBar_ = ui_->mpBar;
  setupHpBar(hpBar_);
  setupMpBar(mpBar_);
}

CharacterCardWidget::~CharacterCardWidget() {
  delete ui_;
}

void CharacterCardWidget::setCharacterName(const QString &name) {
  nameLabel_->setText(name);
}

void CharacterCardWidget::setThumbnail(const QImage &image) {
  thumbnailLabel_->setPixmap(QPixmap::fromImage(image).scaled(128, 96,
                                                       Qt::KeepAspectRatio,
                                                       Qt::SmoothTransformation));
}

void CharacterCardWidget::updateHpMp(int hp, int maxHp, int mp, int maxMp) {
  hpBar_->setRange(0, maxHp);
  hpBar_->setValue(hp);
  hpBar_->setFormat(QString("%1/%2").arg(hp).arg(maxHp));
  mpBar_->setRange(0, maxMp);
  mpBar_->setValue(mp);
  mpBar_->setFormat(QString("%1/%2").arg(mp).arg(maxMp));
}

void CharacterCardWidget::setStateText(const QString &state) {
  stateLabel_->setText(state);
}

void CharacterCardWidget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    emit clicked();
  }
  QFrame::mousePressEvent(event);
}
