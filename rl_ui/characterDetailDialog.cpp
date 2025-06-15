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
#include <QHash>

#include <absl/log/log.h>
#include <algorithm>
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
  cooldownTimer_->setInterval(50);
  connect(cooldownTimer_, &QTimer::timeout, this,
          &CharacterDetailDialog::updateCooldownDisplays);
  cooldownTimer_->start();
}

CharacterDetailDialog::~CharacterDetailDialog() {
  if (cooldownTimer_ != nullptr) {
    cooldownTimer_->stop();
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

  cooldownTimer_->stop();
  ui_->skillCooldownList->clear();
  cooldownItems_.clear();

  QList<SkillCooldown> cooldowns = data.skillCooldowns;
  std::sort(cooldowns.begin(), cooldowns.end(),
            [](const SkillCooldown &a, const SkillCooldown &b) {
              return a.remainingMs > b.remainingMs;
            });

  for (const SkillCooldown &cooldown : cooldowns) {
    QListWidgetItem *item = new QListWidgetItem;

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
        LOG(WARNING) << "Failed to convert skill icon for id " << cooldown.skillId
                     << ": " << ex.what();
      }
    }
    if (!pixmap.isNull()) {
      iconLabel->setPixmap(pixmap.scaled(16, 16));
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
    ci.bar = bar;
    ci.skillName = skillName;
    ci.timer.start();
    cooldownItems_.append(ci);
  }

  if (!cooldownItems_.isEmpty()) {
    cooldownTimer_->start();
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
  for (CooldownItem &item : cooldownItems_) {
    const int elapsed = item.timer.elapsed();
    int remaining = item.startMs - elapsed;
    if (remaining < 0) {
      remaining = 0;
    }
    item.bar->setValue(remaining);
    const double seconds = remaining / 1000.0;
    item.bar->setFormat(QString("%1 (%2s)")
                            .arg(item.skillName)
                            .arg(seconds, 0, 'f', 1));
  }
}
