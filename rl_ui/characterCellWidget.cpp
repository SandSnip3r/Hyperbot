#include "characterCellWidget.hpp"

#include <QLabel>
#include <QProgressBar>
#include <QVBoxLayout>
#include <QMouseEvent>

CharacterCellWidget::CharacterCellWidget(const sro::pk2::GameData &gameData,
                                         QWidget *parent)
    : QFrame(parent), gameData_(gameData) {
  setFrameShape(QFrame::StyledPanel);
  setLineWidth(1);
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->setContentsMargins(4, 4, 4, 4);

  nameLabel_ = new QLabel(this);
  nameLabel_->setAlignment(Qt::AlignHCenter);
  layout->addWidget(nameLabel_);

  hpBar_ = new QProgressBar(this);
  setupHpBar(hpBar_);
  layout->addWidget(hpBar_);

  mpBar_ = new QProgressBar(this);
  setupMpBar(mpBar_);
  layout->addWidget(mpBar_);

  stateLabel_ = new QLabel(this);
  layout->addWidget(stateLabel_);

  detailsWidget_ = new QWidget(this);
  QVBoxLayout *detailLayout = new QVBoxLayout(detailsWidget_);
  detailLayout->setContentsMargins(0, 0, 0, 0);
  stateMachineLabel_ = new QLabel(detailsWidget_);
  stateMachineLabel_->setWordWrap(true);
  detailLayout->addWidget(stateMachineLabel_);
  detailsWidget_->setLayout(detailLayout);
  layout->addWidget(detailsWidget_);

  setLayout(layout);
  setExpanded(false);
}

void CharacterCellWidget::setCharacterName(const QString &name) {
  name_ = name;
  nameLabel_->setText(name_);
}

QString CharacterCellWidget::characterName() const { return name_; }

void CharacterCellWidget::updateCharacterData(const CharacterData &data) {
  hpBar_->setRange(0, data.maxHp);
  hpBar_->setValue(data.currentHp);
  hpBar_->setFormat(QString("%1/%2").arg(data.currentHp).arg(data.maxHp));

  mpBar_->setRange(0, data.maxMp);
  mpBar_->setValue(data.currentMp);
  mpBar_->setFormat(QString("%1/%2").arg(data.currentMp).arg(data.maxMp));

  stateLabel_->setText(data.stateMachine);
  stateMachineLabel_->setText(data.stateMachine);
}

void CharacterCellWidget::setExpanded(bool expanded) {
  expanded_ = expanded;
  detailsWidget_->setVisible(expanded_);
}

void CharacterCellWidget::mousePressEvent(QMouseEvent *event) {
  QFrame::mousePressEvent(event);
  setExpanded(!expanded_);
}
