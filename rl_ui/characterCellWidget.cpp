#include "characterCellWidget.hpp"
#include "barStyles.hpp"

#include <QVBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QMouseEvent>

CharacterCellWidget::CharacterCellWidget(QWidget *parent)
    : QWidget(parent),
      nameLabel_(new QLabel(this)),
      hpBar_(new QProgressBar(this)),
      mpBar_(new QProgressBar(this)),
      stateLabel_(new QLabel(this)),
      detailWidget_(new QWidget(this)) {
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->addWidget(nameLabel_);
  layout->addWidget(hpBar_);
  layout->addWidget(mpBar_);
  layout->addWidget(detailWidget_);

  QVBoxLayout *detailLayout = new QVBoxLayout(detailWidget_);
  detailLayout->addWidget(stateLabel_);
  detailWidget_->setVisible(false);

  setupHpBar(hpBar_);
  setupMpBar(mpBar_);
}

void CharacterCellWidget::setCharacterName(const QString &name) {
  nameLabel_->setText(name);
}

void CharacterCellWidget::updateCharacterData(const CharacterData &data) {
  hpBar_->setRange(0, data.maxHp);
  hpBar_->setValue(data.currentHp);
  hpBar_->setFormat(QString("%1/%2").arg(data.currentHp).arg(data.maxHp));

  mpBar_->setRange(0, data.maxMp);
  mpBar_->setValue(data.currentMp);
  mpBar_->setFormat(QString("%1/%2").arg(data.currentMp).arg(data.maxMp));

  stateLabel_->setText(data.stateMachine);
}

void CharacterCellWidget::mousePressEvent(QMouseEvent *event) {
  Q_UNUSED(event);
  toggleExpanded();
  emit expandRequested(this);
}

void CharacterCellWidget::toggleExpanded() {
  expanded_ = !expanded_;
  detailWidget_->setVisible(expanded_);
}

