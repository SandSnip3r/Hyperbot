#include "characterDetailDialog.hpp"
#include "ui_characterDetailDialog.h"
#include "barStyles.hpp"
#include "textureToQImage.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QListWidget>
#include <QIcon>
#include <QElapsedTimer>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QTimer>
#include <QCoreApplication>
#include <QHash>
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
  cooldownItems_.clear();
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

  QSet<sro::scalar_types::ReferenceSkillId> seen;

  QList<SkillCooldown> cooldowns = data.skillCooldowns;
  std::sort(cooldowns.begin(), cooldowns.end(),
            [](const SkillCooldown &a, const SkillCooldown &b) {
              return a.remainingMs > b.remainingMs;
            });

  for (const SkillCooldown &cooldown : cooldowns) {
    seen.insert(cooldown.skillId);
    const auto &skill =
        gameData_.skillData().getSkillById(cooldown.skillId);
    CooldownItem *item = cooldownItems_.contains(cooldown.skillId)
                             ? &cooldownItems_[cooldown.skillId]
                             : nullptr;

    if (item == nullptr) {
      QListWidgetItem *listItem = new QListWidgetItem;
      QWidget *container = new QWidget;
      QHBoxLayout *layout = new QHBoxLayout(container);
      layout->setContentsMargins(2, 2, 2, 2);

      QLabel *iconLabel = new QLabel(container);
      QPixmap pixmap;
      if (iconCache_.contains(cooldown.skillId)) {
        pixmap = iconCache_.value(cooldown.skillId);
      } else if (const gli::texture2d *texture =
                     gameData_.getSkillIcon(cooldown.skillId)) {
        try {
          QImage img = texture_to_image::texture2dToQImage(*texture);
          pixmap = QPixmap::fromImage(img);
          iconCache_.insert(cooldown.skillId, pixmap);
        } catch (const std::exception &ex) {
          LOG(WARNING) << "Failed to convert skill icon for id "
                       << cooldown.skillId << ": " << ex.what();
        }
      }
      if (!pixmap.isNull()) {
        iconLabel->setPixmap(pixmap.scaled(16, 16));
      }

      QProgressBar *bar = new QProgressBar(container);
      setupCooldownBar(bar);
      bar->setRange(0, skill.actionReuseDelay);
      bar->setValue(cooldown.remainingMs);

      const QString skillName =
          QString::fromStdString(gameData_.getSkillName(cooldown.skillId));
      const double seconds = cooldown.remainingMs / 1000.0;
      bar->setFormat(QString("%1 (%2s)")
                         .arg(skillName)
                         .arg(seconds, 0, 'f', 1));

      layout->addWidget(iconLabel);
      layout->addWidget(bar);

      container->setLayout(layout);
      listItem->setSizeHint(container->sizeHint());
      ui_->skillCooldownList->addItem(listItem);
      ui_->skillCooldownList->setItemWidget(listItem, container);

      CooldownItem ci;
      ci.skillId = cooldown.skillId;
      ci.totalMs = skill.actionReuseDelay;
      ci.startRemainingMs = cooldown.remainingMs;
      ci.bar = bar;
      ci.listItem = listItem;
      ci.container = container;
      ci.skillName = skillName;
      ci.timer.start();
      cooldownItems_.insert(cooldown.skillId, ci);
      item = &cooldownItems_[cooldown.skillId];
    } else {
      int currentRemaining =
          item->startRemainingMs - item->timer.elapsed();
      if (currentRemaining < 0) {
        currentRemaining = 0;
      }
      if (cooldown.remainingMs < currentRemaining) {
        item->startRemainingMs = cooldown.remainingMs;
        item->timer.restart();
        item->totalMs = skill.actionReuseDelay;
        item->bar->setRange(0, skill.actionReuseDelay);
        item->bar->setValue(cooldown.remainingMs);
      }
    }
  }

  for (auto it = cooldownItems_.begin(); it != cooldownItems_.end();) {
    if (!seen.contains(it.key())) {
      delete it.value().listItem;
      it = cooldownItems_.erase(it);
    } else {
      ++it;
    }
  }

  ui_->stateMachineLabel->setText(data.stateMachine);
}

void CharacterDetailDialog::onCharacterDataUpdated(QString name,
                                                   CharacterData data) {
  if (name == name_) {
    updateCharacterData(data);
  }
}

void CharacterDetailDialog::updateCooldownDisplays() {
  QList<CooldownItem *> items;
  items.reserve(cooldownItems_.size());
  for (auto it = cooldownItems_.begin(); it != cooldownItems_.end(); ++it) {
    CooldownItem &item = it.value();
    items.append(&item);
    int remaining = item.startRemainingMs - item.timer.elapsed();
    if (remaining < 0) {
      remaining = 0;
    }
    item.bar->setValue(remaining);
    const double seconds = remaining / 1000.0;
    item.bar->setFormat(QString("%1 (%2s)")
                            .arg(item.skillName)
                            .arg(seconds, 0, 'f', 1));
  }

  std::sort(items.begin(), items.end(), [](CooldownItem *a, CooldownItem *b) {
    int remainingA = a->startRemainingMs - a->timer.elapsed();
    int remainingB = b->startRemainingMs - b->timer.elapsed();
    return remainingA > remainingB;
  });

  for (int i = 0; i < items.size(); ++i) {
    QListWidgetItem *listItem = items[i]->listItem;
    int currentRow = ui_->skillCooldownList->row(listItem);
    if (currentRow != i) {
      ui_->skillCooldownList->takeItem(currentRow);
      ui_->skillCooldownList->insertItem(i, listItem);
      ui_->skillCooldownList->setItemWidget(listItem, items[i]->container);
    }
  }
}
