#include "characterCellWidget.hpp"

#include <QVBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QListWidget>
#include <QMouseEvent>

CharacterCellWidget::CharacterCellWidget(const sro::pk2::GameData &gameData,
                                         QWidget *parent)
    : QWidget(parent), gameData_(gameData) {
  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->setSpacing(2);
  layout->setContentsMargins(2, 2, 2, 2);

  nameLabel_ = new QLabel(this);
  layout->addWidget(nameLabel_, 0, Qt::AlignHCenter);

  hpBar_ = new QProgressBar(this);
  setupHpBar(hpBar_);
  layout->addWidget(hpBar_);

  mpBar_ = new QProgressBar(this);
  setupMpBar(mpBar_);
  layout->addWidget(mpBar_);

  stateLabel_ = new QLabel(this);
  layout->addWidget(stateLabel_, 0, Qt::AlignHCenter);

  detailsContainer_ = new QWidget(this);
  QVBoxLayout *detailLayout = new QVBoxLayout(detailsContainer_);
  detailLayout->setSpacing(2);
  detailLayout->setContentsMargins(0, 0, 0, 0);
  skillCooldownList_ = new QListWidget(detailsContainer_);
  detailLayout->addWidget(skillCooldownList_);
  detailsContainer_->setLayout(detailLayout);
  detailsContainer_->setVisible(false);
  layout->addWidget(detailsContainer_);

  setLayout(layout);
}

void CharacterCellWidget::setCharacterName(const QString &name) {
  name_ = name;
  nameLabel_->setText(name_);
}

void CharacterCellWidget::updateCharacterData(const CharacterData &data) {
  hpBar_->setRange(0, data.maxHp);
  hpBar_->setValue(data.currentHp);
  hpBar_->setFormat(QString("%1/%2").arg(data.currentHp).arg(data.maxHp));

  mpBar_->setRange(0, data.maxMp);
  mpBar_->setValue(data.currentMp);
  mpBar_->setFormat(QString("%1/%2").arg(data.currentMp).arg(data.maxMp));

  stateLabel_->setText(data.stateMachine);

  skillCooldownList_->clear();
  for (const SkillCooldown &cooldown : data.skillCooldowns) {
    QListWidgetItem *item = new QListWidgetItem(
        QString("%1 ms").arg(cooldown.remainingMs), skillCooldownList_);
    item->setData(Qt::UserRole, static_cast<int>(cooldown.remainingMs));
    skillCooldownList_->addItem(item);
  }
}

void CharacterCellWidget::mouseDoubleClickEvent(QMouseEvent *event) {
  Q_UNUSED(event);
  toggleExpanded();
}

void CharacterCellWidget::toggleExpanded() {
  expanded_ = !expanded_;
  detailsContainer_->setVisible(expanded_);
}
