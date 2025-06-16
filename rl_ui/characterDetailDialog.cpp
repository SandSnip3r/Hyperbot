#include "characterDetailDialog.hpp"
#include "ui_characterDetailDialog.h"
#include "barStyles.hpp"
#include "textureToQImage.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QListWidget>
#include <QIcon>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QTimer>
#include <QCoreApplication>
#include <QHash>
#include <QDateTime>
#include <QSet>

#include <absl/log/log.h>
#include <algorithm>
#include <memory>
#include <stdexcept>

QTimer *CharacterDetailDialog::sharedCooldownTimer_ = nullptr;
int CharacterDetailDialog::activeDialogCount_ = 0;

namespace {
} // namespace

CharacterDetailDialog::CharacterDetailDialog(const sro::pk2::GameData &gameData,
                                             QWidget *parent)
    : QDialog(parent), ui_(new Ui::CharacterDetailDialog), gameData_(gameData) {
  ui_->setupUi(this);
  setupHpBar(ui_->hpBar);
  setupMpBar(ui_->mpBar);
  if (sharedCooldownTimer_ == nullptr) {
    sharedCooldownTimer_ = new QTimer(QCoreApplication::instance());
    sharedCooldownTimer_->setInterval(50);
    sharedCooldownTimer_->start();
  }
  connect(sharedCooldownTimer_, &QTimer::timeout, this,
          &CharacterDetailDialog::updateCooldownDisplays);
  ++activeDialogCount_;
}

CharacterDetailDialog::~CharacterDetailDialog() {
  disconnect(sharedCooldownTimer_, &QTimer::timeout, this,
             &CharacterDetailDialog::updateCooldownDisplays);
  --activeDialogCount_;
  if (activeDialogCount_ == 0 && sharedCooldownTimer_ != nullptr) {
    sharedCooldownTimer_->stop();
    sharedCooldownTimer_->deleteLater();
    sharedCooldownTimer_ = nullptr;
  }
  delete ui_;
}

void CharacterDetailDialog::setCharacterName(const QString &name) {
  name_ = name;
  setWindowTitle(name_);
  ui_->nameLabel->setText(name_);
}

void CharacterDetailDialog::updateCharacterData(const CharacterData &data) {
  ui_->hpBar->setRange(0, data.maxHp);
  ui_->hpBar->setValue(data.currentHp);
  ui_->hpBar->setFormat(QString("%1/%2").arg(data.currentHp).arg(data.maxHp));

  ui_->mpBar->setRange(0, data.maxMp);
  ui_->mpBar->setValue(data.currentMp);
  ui_->mpBar->setFormat(QString("%1/%2").arg(data.currentMp).arg(data.maxMp));

  const qint64 now = QDateTime::currentMSecsSinceEpoch();

  QSet<sro::scalar_types::ReferenceSkillId> incomingIds;
  QList<SkillCooldown> cooldowns = data.skillCooldowns;
  std::sort(cooldowns.begin(), cooldowns.end(),
            [](const SkillCooldown &a, const SkillCooldown &b) {
              return a.remainingMs > b.remainingMs;
            });

  for (const SkillCooldown &cooldown : cooldowns) {
    incomingIds.insert(cooldown.skillId);

    if (!gameData_.skillData().haveSkillWithId(cooldown.skillId)) {
      LOG(WARNING) << "Unknown skill id " << cooldown.skillId
                   << " received in cooldown update";
      continue;
    }

    const sro::pk2::ref::Skill &skill =
        gameData_.skillData().getSkillById(cooldown.skillId);

    int predicted = cooldown.remainingMs -
                    static_cast<int>(now - cooldown.timestampMs);
    if (predicted < 0) {
      predicted = 0;
    }
    if (predicted > skill.actionReuseDelay) {
      predicted = skill.actionReuseDelay;
    }

    CooldownItem *ci = nullptr;
    if (cooldownItems_.contains(cooldown.skillId)) {
      ci = &cooldownItems_[cooldown.skillId];
      int currentRemaining =
          std::max(0, ci->remainingMs - static_cast<int>(now - ci->timestampMs));
      ci->remainingMs = std::min(currentRemaining, predicted);
      ci->timestampMs = now;
      ci->totalMs = skill.actionReuseDelay;
    } else {
      QListWidgetItem *item = new QListWidgetItem;
      QWidget *container = new QWidget;
      QHBoxLayout *layout = new QHBoxLayout(container);
      layout->setContentsMargins(2, 2, 2, 2);

      QLabel *iconLabel = new QLabel(container);
      QPixmap pixmap = getIconForSkillId(cooldown.skillId);
      if (!pixmap.isNull()) {
        iconLabel->setPixmap(pixmap.scaled(16, 16));
      }

      QProgressBar *bar = new QProgressBar(container);
      setupCooldownBar(bar);
      bar->setRange(0, skill.actionReuseDelay);

      layout->addWidget(iconLabel);
      layout->addWidget(bar);
      container->setLayout(layout);
      item->setSizeHint(container->sizeHint());
      ui_->skillCooldownList->addItem(item);
      ui_->skillCooldownList->setItemWidget(item, container);

      CooldownItem newItem;
      newItem.skillId = cooldown.skillId;
      newItem.totalMs = skill.actionReuseDelay;
      newItem.remainingMs = predicted;
      newItem.timestampMs = now;
      newItem.item = item;
      newItem.container = container;
      newItem.bar = bar;
      newItem.skillName =
          QString::fromStdString(gameData_.getSkillName(cooldown.skillId));
      cooldownItems_.insert(cooldown.skillId, newItem);
      ci = &cooldownItems_[cooldown.skillId];
    }

    ci->bar->setRange(0, skill.actionReuseDelay);
    ci->bar->setValue(ci->remainingMs);
    const double seconds = ci->remainingMs / 1000.0;
    ci->bar->setFormat(QString("%1 (%2s)")
                           .arg(ci->skillName)
                           .arg(seconds, 0, 'f', 1));
    ci->item->setData(Qt::UserRole, ci->remainingMs);
  }

  for (auto it = cooldownItems_.begin(); it != cooldownItems_.end();) {
    if (!incomingIds.contains(it.key())) {
      int row = ui_->skillCooldownList->row(it->item);
      delete ui_->skillCooldownList->takeItem(row);
      it = cooldownItems_.erase(it);
    } else {
      ++it;
    }
  }

  ui_->skillCooldownList->sortItems(Qt::DescendingOrder);

  ui_->stateMachineLabel->setText(data.stateMachine);
}

void CharacterDetailDialog::onCharacterDataUpdated(QString name,
                                                   CharacterData data) {
  if (name == name_) {
    updateCharacterData(data);
  }
}

void CharacterDetailDialog::updateCooldownDisplays() {
  const qint64 now = QDateTime::currentMSecsSinceEpoch();
  for (auto it = cooldownItems_.begin(); it != cooldownItems_.end(); ++it) {
    CooldownItem &item = it.value();
    int remaining =
        std::max(0, item.remainingMs - static_cast<int>(now - item.timestampMs));
    item.bar->setValue(remaining);
    const double seconds = remaining / 1000.0;
    item.bar->setFormat(QString("%1 (%2s)")
                            .arg(item.skillName)
                            .arg(seconds, 0, 'f', 1));
  }
}

QPixmap CharacterDetailDialog::getIconForSkillId(
    sro::scalar_types::ReferenceSkillId skillId) {
  if (iconCache_.contains(skillId)) {
    return iconCache_.value(skillId);
  }
  const gli::texture2d *texture = gameData_.getSkillIcon(skillId);
  if (texture == nullptr) {
    return QPixmap();
  }
  try {
    QImage img = texture_to_image::texture2dToQImage(*texture);
    QPixmap pixmap = QPixmap::fromImage(img);
    iconCache_.insert(skillId, pixmap);
    return pixmap;
  } catch (const std::exception &ex) {
    LOG(WARNING) << "Failed to convert skill icon for id " << skillId << ": "
                 << ex.what();
    return QPixmap();
  }
}
