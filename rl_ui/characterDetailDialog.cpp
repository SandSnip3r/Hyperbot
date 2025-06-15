#include "characterDetailDialog.hpp"
#include "ui_characterDetailDialog.h"
#include "barStyles.hpp"
#include "textureToQImage.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QListWidget>
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

  ui_->skillCooldownList->clear();
  for (const SkillCooldown &cooldown : data.skillCooldowns) {
    QListWidgetItem *item = new QListWidgetItem;
    if (const gli::texture2d *texture = gameData_.getSkillIcon(cooldown.skillId)) {
      try {
        const QImage img = texture_to_image::texture2dToQImage(*texture);
        item->setIcon(QIcon(QPixmap::fromImage(img)));
      } catch (const std::exception &ex) {
        LOG(WARNING) << "Failed to convert skill icon for id " << cooldown.skillId
                     << ": " << ex.what();
      }
    }
    const double seconds = cooldown.remainingMs / 1000.0;
    item->setText(QString::number(seconds, 'f', 1));
    ui_->skillCooldownList->addItem(item);
  }

  ui_->stateMachineLabel->setText(data.stateMachine);
}

void CharacterDetailDialog::onCharacterDataUpdated(QString name,
                                                   CharacterData data) {
  if (name == name_) {
    updateCharacterData(data);
  }
}
