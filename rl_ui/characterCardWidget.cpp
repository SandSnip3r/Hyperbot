#include "characterCardWidget.hpp"
#include "barStyles.hpp"

#include <QHBoxLayout>
#include <QMouseEvent>
#include <QVBoxLayout>

CharacterCardWidget::CharacterCardWidget(QWidget *parent)
    : QFrame(parent),
      thumbnailLabel_(new QLabel(this)),
      nameLabel_(new QLabel(this)),
      stateLabel_(new QLabel(this)),
      hpBar_(new QProgressBar(this)),
      mpBar_(new QProgressBar(this)) {
  setFrameShape(QFrame::StyledPanel);
  setFrameShadow(QFrame::Raised);

  thumbnailLabel_->setFixedSize(120, 90);
  thumbnailLabel_->setStyleSheet("background-color: black;");

  setupHpBar(hpBar_);
  setupMpBar(mpBar_);

  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->addWidget(thumbnailLabel_, 0, Qt::AlignCenter);
  layout->addWidget(nameLabel_, 0, Qt::AlignCenter);
  layout->addWidget(hpBar_);
  layout->addWidget(mpBar_);
  layout->addWidget(stateLabel_, 0, Qt::AlignCenter);
  setLayout(layout);
}

void CharacterCardWidget::setCharacterName(const QString &name) {
  name_ = name;
  nameLabel_->setText(name);
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

void CharacterCardWidget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    emit cardClicked(name_);
  }
  QFrame::mousePressEvent(event);
}
