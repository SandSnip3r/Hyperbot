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
#include <QHeaderView>
#include <limits>

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
  qValuesTable_ = new QTableWidget(this);
  qValuesTable_->setColumnCount(2);
  QStringList headers;
  headers << "Action" << "Q-Value";
  qValuesTable_->setHorizontalHeaderLabels(headers);
  qValuesTable_->verticalHeader()->setDefaultSectionSize(18);
  ui_->verticalLayout->addWidget(qValuesTable_);
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

void CharacterDetailDialog::updateHpMp(int currentHp, int maxHp, int currentMp,
                                       int maxMp) {
  ui_->hpBar->setRange(0, maxHp);
  ui_->hpBar->setValue(currentHp);
  ui_->hpBar->setFormat(QString("%1/%2").arg(currentHp).arg(maxHp));

  ui_->mpBar->setRange(0, maxMp);
  ui_->mpBar->setValue(currentMp);
  ui_->mpBar->setFormat(QString("%1/%2").arg(currentMp).arg(maxMp));
}

void CharacterDetailDialog::updateStateMachine(const QString &stateMachine) {
  ui_->stateMachineLabel->setText(stateMachine);
}

void CharacterDetailDialog::updateSkillCooldowns(
    const QList<SkillCooldown> &cooldowns) {
  const qint64 now = QDateTime::currentMSecsSinceEpoch();

  QSet<sro::scalar_types::ReferenceSkillId> incomingIds;

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
      CooldownListItem *item = new CooldownListItem;
      QWidget *container = new QWidget;
      QHBoxLayout *layout = new QHBoxLayout(container);
      layout->setContentsMargins(2, 2, 2, 2);

      QLabel *iconLabel = new QLabel(container);
      QPixmap pixmap = getIconForSkillId(cooldown.skillId);
      if (!pixmap.isNull()) {
        iconLabel->setPixmap(pixmap.scaled(24, 24));
      }

      QProgressBar *bar = new QProgressBar(container);
      setupCooldownBar(bar);
      bar->setRange(0, skill.actionReuseDelay);

      layout->addWidget(iconLabel);
      layout->addWidget(bar);
      container->setLayout(layout);
      item->setSizeHint(container->sizeHint());
      item->setData(Qt::UserRole, predicted);
      item->setText("");
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
    ci->item->setText("");
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
}

void CharacterDetailDialog::updateQValues(const QVector<float> &qValues) {
  if (qValuesTable_->rowCount() != qValues.size()) {
    qValuesTable_->clearContents();
    qValuesTable_->setRowCount(qValues.size());
    qValueBars_.clear();
    for (int i = 0; i < qValues.size(); ++i) {
      QTableWidgetItem *indexItem = new QTableWidgetItem(QString::number(i));
      qValuesTable_->setItem(i, 0, indexItem);
      QProgressBar *bar = new QProgressBar;
      bar->setRange(0, 100);
      qValuesTable_->setCellWidget(i, 1, bar);
      qValuesTable_->setRowHeight(i, 18);
      qValueBars_.append(bar);
    }
  }

  float maxVal = -std::numeric_limits<float>::infinity();
  for (float v : qValues) {
    if (v != -std::numeric_limits<float>::infinity() && v > maxVal) {
      maxVal = v;
    }
  }
  if (maxVal == -std::numeric_limits<float>::infinity()) {
    maxVal = 0.0f;
  }

  for (int i = 0; i < qValues.size(); ++i) {
    QProgressBar *bar = qValueBars_.value(i);
    if (!bar) {
      continue;
    }
    if (qValues[i] == -std::numeric_limits<float>::infinity()) {
      bar->setValue(0);
      bar->setFormat("-inf");
    } else if (maxVal == 0.0f) {
      bar->setValue(100);
      bar->setFormat(QString::number(qValues[i], 'f', 2));
    } else {
      int pct = static_cast<int>((qValues[i] / maxVal) * 100.0f);
      if (pct < 0) pct = 0;
      bar->setValue(pct);
      bar->setFormat(QString::number(qValues[i], 'f', 2));
    }
  }
}

void CharacterDetailDialog::updateCharacterData(const CharacterData &data) {
  updateHpMp(data.currentHp, data.maxHp, data.currentMp, data.maxMp);
  updateSkillCooldowns(data.skillCooldowns);
  updateQValues(data.qValues);
  updateStateMachine(data.stateMachine);
}

void CharacterDetailDialog::onCharacterDataUpdated(QString name,
                                                   CharacterData data) {
  if (name == name_) {
    updateCharacterData(data);
  }
}

void CharacterDetailDialog::onCharacterStatusUpdated(QString name, int currentHp,
                                                     int maxHp, int currentMp,
                                                     int maxMp) {
  if (name == name_) {
    updateHpMp(currentHp, maxHp, currentMp, maxMp);
  }
}

void CharacterDetailDialog::onActiveStateMachineUpdated(QString name,
                                                        QString stateMachine) {
  if (name == name_) {
    updateStateMachine(stateMachine);
  }
}

void CharacterDetailDialog::onSkillCooldownsUpdated(QString name,
                                                    QList<SkillCooldown> cooldowns) {
  if (name == name_) {
    updateSkillCooldowns(cooldowns);
  }
}

void CharacterDetailDialog::onQValuesUpdated(QString name,
                                             QVector<float> qValues) {
  if (name == name_) {
    updateQValues(qValues);
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
