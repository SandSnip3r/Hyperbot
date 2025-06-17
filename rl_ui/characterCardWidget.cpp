#include "characterCardWidget.hpp"

#include <QVBoxLayout>
#include <QMouseEvent>

CharacterCardWidget::CharacterCardWidget(QWidget *parent)
    : QFrame(parent),
      thumbnailLabel_(new QLabel(this)),
      nameLabel_(new QLabel(this)),
      stateLabel_(new QLabel(this)),
      hpBar_(new QProgressBar(this)),
      mpBar_(new QProgressBar(this)) {
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->setContentsMargins(2, 2, 2, 2);
  layout->addWidget(thumbnailLabel_);
  layout->addWidget(nameLabel_);
  layout->addWidget(hpBar_);
  layout->addWidget(mpBar_);
  layout->addWidget(stateLabel_);
  setLayout(layout);
  setupHpBar(hpBar_);
  setupMpBar(mpBar_);
  thumbnailLabel_->setAlignment(Qt::AlignCenter);
  thumbnailLabel_->setFixedSize(100, 60);
  setFrameShape(QFrame::StyledPanel);
  setFrameShadow(QFrame::Raised);
}

void CharacterCardWidget::setCharacterName(const QString &name) {
  name_ = name;
  nameLabel_->setText(name);
}

void CharacterCardWidget::setPairColor(const QColor &color) {
  setStyleSheet(QString("border: 2px solid %1;").arg(color.name()));
}

void CharacterCardWidget::updateCharacterData(const CharacterData &data) {
  hpBar_->setRange(0, data.maxHp);
  hpBar_->setValue(data.currentHp);
  hpBar_->setFormat(QString("%1/%2").arg(data.currentHp).arg(data.maxHp));
  mpBar_->setRange(0, data.maxMp);
  mpBar_->setValue(data.currentMp);
  mpBar_->setFormat(QString("%1/%2").arg(data.currentMp).arg(data.maxMp));
  stateLabel_->setText(data.stateMachine);
}

void CharacterCardWidget::setThumbnail(const QPixmap &pixmap) {
  if (pixmap.isNull()) {
    thumbnailLabel_->clear();
    return;
  }
  thumbnailLabel_->setPixmap(pixmap.scaled(thumbnailLabel_->size(),
                                           Qt::KeepAspectRatio,
                                           Qt::SmoothTransformation));
}

void CharacterCardWidget::mousePressEvent(QMouseEvent *event) {
  QFrame::mousePressEvent(event);
  emit clicked(name_);
}
