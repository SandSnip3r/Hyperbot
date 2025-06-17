#include "characterCardWidget.hpp"
#include "barStyles.hpp"

#include <QVBoxLayout>
#include <QMouseEvent>
#include <QPalette>

CharacterCardWidget::CharacterCardWidget(const QString &name, QWidget *parent)
    : QFrame(parent), thumbnailLabel_(new QLabel), hpBar_(new QProgressBar),
      mpBar_(new QProgressBar), stateLabel_(new QLabel), name_(name) {
  setFrameStyle(QFrame::Box | QFrame::Plain);
  setLineWidth(2);
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->setContentsMargins(2, 2, 2, 2);
  layout->setSpacing(2);

  thumbnailLabel_->setFixedSize(120, 90);
  thumbnailLabel_->setAlignment(Qt::AlignCenter);
  layout->addWidget(thumbnailLabel_);

  setupHpBar(hpBar_);
  layout->addWidget(hpBar_);
  setupMpBar(mpBar_);
  layout->addWidget(mpBar_);

  stateLabel_->setAlignment(Qt::AlignCenter);
  layout->addWidget(stateLabel_);

  setName(name);
}

void CharacterCardWidget::setName(const QString &name) {
  name_ = name;
  setToolTip(name_);
}

void CharacterCardWidget::updateStatus(int currentHp, int maxHp, int currentMp,
                                       int maxMp) {
  hpBar_->setRange(0, maxHp);
  hpBar_->setValue(currentHp);
  hpBar_->setFormat(QString("%1/%2").arg(currentHp).arg(maxHp));
  mpBar_->setRange(0, maxMp);
  mpBar_->setValue(currentMp);
  mpBar_->setFormat(QString("%1/%2").arg(currentMp).arg(maxMp));
}

void CharacterCardWidget::setState(const QString &state) {
  stateLabel_->setText(state);
}

void CharacterCardWidget::setPairColor(const QColor &color) {
  QPalette pal = palette();
  pal.setColor(QPalette::WindowText, color);
  setStyleSheet(QString("QFrame { border: 2px solid %1; }")
                    .arg(color.name()));
}

void CharacterCardWidget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    emit clicked(name_);
  }
  QFrame::mousePressEvent(event);
}
