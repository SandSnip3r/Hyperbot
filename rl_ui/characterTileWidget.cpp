#include "characterTileWidget.hpp"
#include "barStyles.hpp"
#include "textureToQImage.hpp"

#include <QSet>
#include <QMouseEvent>
#include <QIcon>
#include <QSize>
#include <QImage>
#include <QPixmap>
#include <exception>
#include <QDateTime>
#include <absl/log/log.h>
#include <algorithm>

CharacterTileWidget::CharacterTileWidget(const sro::pk2::GameData &gameData,
                                         QWidget *parent)
    : QFrame(parent), gameData_(gameData) {
  setFrameShape(QFrame::StyledPanel);
  layout_ = new QVBoxLayout(this);
  layout_->setContentsMargins(2, 2, 2, 2);
  nameLabel_ = new QLabel(this);
  nameLabel_->setAlignment(Qt::AlignHCenter);
  hpBar_ = new QProgressBar(this);
  mpBar_ = new QProgressBar(this);
  setupHpBar(hpBar_);
  setupMpBar(mpBar_);
  cooldownList_ = new QListWidget(this);
  cooldownList_->setIconSize(QSize(24, 24));
  stateLabel_ = new QLabel(this);
  stateLabel_->setWordWrap(true);

  layout_->addWidget(nameLabel_);
  layout_->addWidget(hpBar_);
  layout_->addWidget(mpBar_);
  layout_->addWidget(cooldownList_);
  layout_->addWidget(stateLabel_);

  updateDetailVisibility();
}

void CharacterTileWidget::setCharacterName(const QString &name) {
  nameLabel_->setText(name);
}

void CharacterTileWidget::updateCharacterData(const CharacterData &data) {
  hpBar_->setRange(0, data.maxHp);
  hpBar_->setValue(data.currentHp);
  hpBar_->setFormat(QString("%1/%2").arg(data.currentHp).arg(data.maxHp));

  mpBar_->setRange(0, data.maxMp);
  mpBar_->setValue(data.currentMp);
  mpBar_->setFormat(QString("%1/%2").arg(data.currentMp).arg(data.maxMp));

  stateLabel_->setText(data.stateMachine);

  const qint64 now = QDateTime::currentMSecsSinceEpoch();
  QSet<sro::scalar_types::ReferenceSkillId> incomingIds;
  for (const SkillCooldown &cooldown : data.skillCooldowns) {
    incomingIds.insert(cooldown.skillId);
    int predicted = cooldown.remainingMs - static_cast<int>(now - cooldown.timestampMs);
    if (predicted < 0) {
      predicted = 0;
    }
    const sro::pk2::ref::Skill &skill =
        gameData_.skillData().getSkillById(cooldown.skillId);
    if (predicted > skill.actionReuseDelay) {
      predicted = skill.actionReuseDelay;
    }
    QListWidgetItem *item = nullptr;
    for (int i = 0; i < cooldownList_->count(); ++i) {
      QListWidgetItem *it = cooldownList_->item(i);
      if (it->data(Qt::UserRole).toInt() == static_cast<int>(cooldown.skillId)) {
        item = it;
        break;
      }
    }
    if (!item) {
      item = new QListWidgetItem(cooldownList_);
      const gli::texture2d *texture = gameData_.getSkillIcon(cooldown.skillId);
      if (texture) {
        try {
          QImage img = texture_to_image::texture2dToQImage(*texture);
          QPixmap pixmap = QPixmap::fromImage(img);
          item->setIcon(QIcon(pixmap));
        } catch (const std::exception &ex) {
          LOG(WARNING) << "Failed to convert skill icon for id " << cooldown.skillId
                       << ": " << ex.what();
        }
      }
      cooldownList_->addItem(item);
    }
    item->setData(Qt::UserRole, static_cast<int>(cooldown.skillId));
    item->setText(
        QString("%1 (%2s)")
            .arg(QString::fromStdString(gameData_.getSkillName(cooldown.skillId)))
            .arg(predicted / 1000.0, 0, 'f', 1));
  }

  for (int i = cooldownList_->count() - 1; i >= 0; --i) {
    QListWidgetItem *it = cooldownList_->item(i);
    sro::scalar_types::ReferenceSkillId id =
        static_cast<sro::scalar_types::ReferenceSkillId>(it->data(Qt::UserRole).toInt());
    if (!incomingIds.contains(id)) {
      delete cooldownList_->takeItem(i);
    }
  }
}

void CharacterTileWidget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    expanded_ = !expanded_;
    updateDetailVisibility();
    emit expandedChanged(expanded_);
  }
  QFrame::mousePressEvent(event);
}

void CharacterTileWidget::updateDetailVisibility() {
  cooldownList_->setVisible(expanded_);
  stateLabel_->setVisible(expanded_);
}

