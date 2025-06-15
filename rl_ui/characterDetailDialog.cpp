#include "characterDetailDialog.hpp"
#include "ui_characterDetailDialog.h"
#include "barStyles.hpp"
#include "textureToQImage.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QElapsedTimer>
#include <QHash>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QListWidget>
#include <QPixmap>
#include <QProgressBar>
#include <QTimer>
#include <QSet>

#include <absl/log/log.h>
#include <memory>
#include <stdexcept>

namespace {
} // namespace

CharacterDetailDialog::CharacterDetailDialog(const sro::pk2::GameData &gameData,
                                             QWidget *parent)
    : QDialog(parent), ui_(new Ui::CharacterDetailDialog), gameData_(gameData) {
  ui_->setupUi(this);
  setupHpBar(ui_->hpBar);
  setupMpBar(ui_->mpBar);
  cooldownTimer_ = new QTimer(this);
  cooldownTimer_->setInterval(100);
  connect(cooldownTimer_, &QTimer::timeout, this,
          &CharacterDetailDialog::updateCooldownDisplays);
  cooldownTimer_->start();
}

CharacterDetailDialog::~CharacterDetailDialog() {
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
  QSet<sro::scalar_types::ReferenceSkillId> updatedIds;

  for (const SkillCooldown &cooldown : data.skillCooldowns) {
    updatedIds.insert(cooldown.skillId);
    auto it = std::find_if(cooldownItems_.begin(), cooldownItems_.end(),
                           [&](const CooldownItem &ci) {
                             return ci.skillId == cooldown.skillId;
                           });
    if (it == cooldownItems_.end()) {
      QListWidgetItem *item = new QListWidgetItem;

      QWidget *container = new QWidget(ui_->skillCooldownList);
      QHBoxLayout *layout = new QHBoxLayout(container);
      layout->setContentsMargins(2, 2, 2, 2);

      QLabel *iconLabel = new QLabel(container);
      const QPixmap pix = getSkillPixmap(cooldown.skillId);
      if (!pix.isNull()) {
        iconLabel->setPixmap(pix);
      }

      QProgressBar *bar = new QProgressBar(container);
      bar->setRange(0, cooldown.remainingMs);
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
      item->setSizeHint(container->sizeHint());
      ui_->skillCooldownList->addItem(item);
      ui_->skillCooldownList->setItemWidget(item, container);

      CooldownItem ci;
      ci.skillId = cooldown.skillId;
      ci.startMs = cooldown.remainingMs;
      ci.item = item;
      ci.container = container;
      ci.bar = bar;
      ci.skillName = skillName;
      ci.timer.start();
      cooldownItems_.append(ci);
    } else {
      it->startMs = cooldown.remainingMs;
      it->timer.restart();
      it->bar->setRange(0, cooldown.remainingMs);
    }
  }

  for (int i = cooldownItems_.size() - 1; i >= 0; --i) {
    if (!updatedIds.contains(cooldownItems_[i].skillId)) {
      QWidget *widget = ui_->skillCooldownList->itemWidget(cooldownItems_[i].item);
      delete widget;
      delete ui_->skillCooldownList->takeItem(ui_->skillCooldownList->row(cooldownItems_[i].item));
      cooldownItems_.removeAt(i);
    }
  }

  sortCooldowns();

  ui_->stateMachineLabel->setText(data.stateMachine);
}

void CharacterDetailDialog::onCharacterDataUpdated(QString name,
                                                   CharacterData data) {
  if (name == name_) {
    updateCharacterData(data);
  }
}

void CharacterDetailDialog::updateCooldownDisplays() {
  for (CooldownItem &item : cooldownItems_) {
    int remaining = item.remainingMs();
    if (remaining < 0) {
      remaining = 0;
    }
    item.bar->setValue(remaining);
    const double seconds = remaining / 1000.0;
    item.bar->setFormat(QString("%1 (%2s)")
                            .arg(item.skillName)
                            .arg(seconds, 0, 'f', 1));
  }

  sortCooldowns();
}

QPixmap CharacterDetailDialog::getSkillPixmap(
    sro::scalar_types::ReferenceSkillId id) {
  auto it = iconCache_.find(id);
  if (it != iconCache_.end()) {
    return it.value();
  }

  if (const gli::texture2d *texture = gameData_.getSkillIcon(id)) {
    try {
      const QImage img = texture_to_image::texture2dToQImage(*texture);
      QPixmap pix = QPixmap::fromImage(img).scaled(16, 16);
      iconCache_.insert(id, pix);
      return pix;
    } catch (const std::exception &ex) {
      LOG(WARNING) << "Failed to convert skill icon for id " << id << ": "
                   << ex.what();
    }
  }

  return QPixmap();
}

void CharacterDetailDialog::sortCooldowns() {
  std::sort(cooldownItems_.begin(), cooldownItems_.end(),
            [](const CooldownItem &a, const CooldownItem &b) {
              return a.remainingMs() > b.remainingMs();
            });

  for (int i = 0; i < cooldownItems_.size(); ++i) {
    CooldownItem &ci = cooldownItems_[i];
    ui_->skillCooldownList->takeItem(ui_->skillCooldownList->row(ci.item));
    ui_->skillCooldownList->insertItem(i, ci.item);
    ui_->skillCooldownList->setItemWidget(ci.item, ci.container);
  }
}
