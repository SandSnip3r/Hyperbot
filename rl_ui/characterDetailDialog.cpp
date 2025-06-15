#include "characterDetailDialog.hpp"
#include "ui_characterDetailDialog.h"
#include "barStyles.hpp"
#include "textureToQImage.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QListWidget>
#include <QHBoxLayout>
#include <QLabel>
#include <QIcon>

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
  updateTimer_.setInterval(100);
  connect(&updateTimer_, &QTimer::timeout, this,
          &CharacterDetailDialog::updateCooldownBars);
  updateTimer_.start();
}

CharacterDetailDialog::~CharacterDetailDialog() {
  updateTimer_.stop();
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

  // Clear existing cooldown widgets.
  for (auto &widget : cooldownWidgets_) {
    delete widget.item;
  }
  cooldownWidgets_.clear();
  ui_->skillCooldownList->clear();

  for (const SkillCooldown &cooldown : data.skillCooldowns) {
    QListWidgetItem *item = new QListWidgetItem;
    QWidget *widget = new QWidget;
    QHBoxLayout *layout = new QHBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    QLabel *iconLabel = new QLabel;
    iconLabel->setFixedSize(32, 32);
    if (const gli::texture2d *texture = gameData_.getSkillIcon(cooldown.skillId)) {
      try {
        const QImage img = texture_to_image::texture2dToQImage(*texture);
        iconLabel->setPixmap(QPixmap::fromImage(img));
      } catch (const std::exception &ex) {
        LOG(WARNING) << "Failed to convert skill icon for id " << cooldown.skillId
                     << ": " << ex.what();
      }
    }
    QProgressBar *bar = new QProgressBar;
    bar->setRange(0, cooldown.remainingMs);
    bar->setValue(cooldown.remainingMs);
    bar->setFormat(QString("%1s")
                       .arg(cooldown.remainingMs / 1000.0, 0, 'f', 2));
    layout->addWidget(iconLabel);
    layout->addWidget(bar);
    item->setSizeHint(widget->sizeHint());
    ui_->skillCooldownList->addItem(item);
    ui_->skillCooldownList->setItemWidget(item, widget);

    CooldownWidget cw;
    cw.item = item;
    cw.bar = bar;
    cw.totalMs = cooldown.remainingMs;
    cw.remainingMs = cooldown.remainingMs;
    cooldownWidgets_[cooldown.skillId] = cw;
  }

  ui_->stateMachineLabel->setText(data.stateMachine);
}

void CharacterDetailDialog::onCharacterDataUpdated(QString name,
                                                   CharacterData data) {
  if (name == name_) {
    updateCharacterData(data);
  }
}

void CharacterDetailDialog::updateCooldownBars() {
  const double delta = updateTimer_.interval();
  for (auto it = cooldownWidgets_.begin(); it != cooldownWidgets_.end(); ++it) {
    it->remainingMs -= delta;
    if (it->remainingMs < 0) {
      it->remainingMs = 0;
    }
    it->bar->setValue(static_cast<int>(it->remainingMs));
    it->bar->setFormat(QString("%1s")
                           .arg(it->remainingMs / 1000.0, 0, 'f', 2));
  }
}
