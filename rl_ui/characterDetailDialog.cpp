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
#include <QPainter>
#include <limits>

#include <absl/log/log.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <array>
#include <cstddef>

QTimer *CharacterDetailDialog::sharedCooldownTimer_ = nullptr;
int CharacterDetailDialog::activeDialogCount_ = 0;

namespace {
enum class ActionType { Text, Skill, Item };

struct ActionInfo {
  ActionType type;
  const char *text;
  sro::scalar_types::ReferenceSkillId skillId;
  sro::scalar_types::ReferenceObjectId itemId;
};

const ActionInfo kActionInfos[] = {
    {ActionType::Text, "Sleep", 0, 0},                     // 0
    {ActionType::Text, "Attack", 0, 0},                    // 1
    // {ActionType::Skill, nullptr, 28, 0},                   // 2
    // {ActionType::Skill, nullptr, 131, 0},                  // 3
    // {ActionType::Skill, nullptr, 554, 0},                  // 4
    // {ActionType::Skill, nullptr, 1253, 0},                 // 5
    // {ActionType::Skill, nullptr, 1256, 0},                 // 6
    // {ActionType::Skill, nullptr, 1271, 0},                 // 7
    // {ActionType::Skill, nullptr, 1272, 0},                 // 8
    // {ActionType::Skill, nullptr, 1281, 0},                 // 9
    // {ActionType::Skill, nullptr, 1335, 0},                 // 10
    // {ActionType::Skill, nullptr, 1377, 0},                 // 11
    // {ActionType::Skill, nullptr, 1380, 0},                 // 12
    // {ActionType::Skill, nullptr, 1398, 0},                 // 13
    // {ActionType::Skill, nullptr, 1399, 0},                 // 14
    // {ActionType::Skill, nullptr, 1410, 0},                 // 15
    // {ActionType::Skill, nullptr, 1421, 0},                 // 16
    // {ActionType::Skill, nullptr, 1441, 0},                 // 17
    // {ActionType::Skill, nullptr, 30577, 0},                // 18
    // {ActionType::Skill, nullptr, 37, 0},                   // 19
    // {ActionType::Skill, nullptr, 114, 0},                  // 20
    // {ActionType::Skill, nullptr, 298, 0},                  // 21
    // {ActionType::Skill, nullptr, 300, 0},                  // 22 // Stab Smash
    // {ActionType::Skill, nullptr, 322, 0},                  // 23
    // {ActionType::Skill, nullptr, 339, 0},                  // 24
    // {ActionType::Skill, nullptr, 371, 0},                  // 25
    // {ActionType::Skill, nullptr, 588, 0},                  // 26 // Soul Cut Blade
    // {ActionType::Skill, nullptr, 610, 0},                  // 27
    // {ActionType::Skill, nullptr, 644, 0},                  // 28
    // {ActionType::Skill, nullptr, 1315, 0},                 // 29
    // {ActionType::Skill, nullptr, 1343, 0},                 // 30
    // {ActionType::Skill, nullptr, 1449, 0},                 // 31
    {ActionType::Item, nullptr, 0, 5},                     // 32
    // {ActionType::Item, nullptr, 0, 12},                    // 33
    // {ActionType::Item, nullptr, 0, 56},                    // 34
};
} // namespace

CharacterDetailDialog::CharacterDetailDialog(const sro::pk2::GameData &gameData,
                                             QWidget *parent)
    : QDialog(parent), ui_(new Ui::CharacterDetailDialog), gameData_(gameData) {
  ui_->setupUi(this);
  hpPotionIconLabel_ = ui_->hpPotionIcon;
  mpPotionIconLabel_ = ui_->mpPotionIcon;
  updateItemCount(static_cast<sro::scalar_types::ReferenceObjectId>(5), 0);
  updateItemCount(static_cast<sro::scalar_types::ReferenceObjectId>(12), 0);
  setupHpBar(ui_->hpBar);
  setupMpBar(ui_->mpBar);
  ui_->qValuesTable->setColumnCount(2);
  QStringList headers;
  headers << "Action" << "Q-Value";
  ui_->qValuesTable->setHorizontalHeaderLabels(headers);
  ui_->qValuesTable->verticalHeader()->setDefaultSectionSize(20);
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

void CharacterDetailDialog::updateItemCount(sro::scalar_types::ReferenceObjectId itemRefId, int count) {
  const int iconSize = 30;
  const int fontSize = 9;

  QLabel *label = nullptr;
  if (itemRefId == 5) {
    label = hpPotionIconLabel_;
  } else if (itemRefId == 12) {
    label = mpPotionIconLabel_;
  } else {
    return;
  }

  QPixmap base = getIconForItemId(itemRefId);
  if (base.isNull()) {
    label->clear();
    return;
  }
  QPixmap pm = base.scaled(iconSize, iconSize, Qt::KeepAspectRatio,
                           Qt::SmoothTransformation);
  QPainter painter(&pm);
  painter.setPen(Qt::white);
  QFont font = painter.font();
  font.setPixelSize(fontSize);
  painter.setFont(font);
  painter.drawText(QPoint(2, fontSize), QString::number(count));
  painter.end();
  label->setPixmap(pm);
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
  if (ui_->qValuesTable->rowCount() != qValues.size()) {
    ui_->qValuesTable->clearContents();
    ui_->qValuesTable->setRowCount(qValues.size());
    qValueBars_.clear();
    for (int i = 0; i < qValues.size(); ++i) {
      QTableWidgetItem *indexItem = new QTableWidgetItem;
      if (i < static_cast<int>(std::size(kActionInfos))) {
        const ActionInfo &info = kActionInfos[i];
        if (info.type == ActionType::Text) {
          indexItem->setText(QString::fromLatin1(info.text));
        } else if (info.type == ActionType::Skill) {
          QPixmap pm = getIconForSkillId(info.skillId);
          if (!pm.isNull()) indexItem->setData(Qt::DecorationRole, pm.scaled(24, 24));
          QString toolTip = QString("%1 (%2)")
                                .arg(QString::fromStdString(gameData_.getSkillName(info.skillId)))
                                .arg(info.skillId);
          indexItem->setToolTip(toolTip);
        } else if (info.type == ActionType::Item) {
          QPixmap pm = getIconForItemId(info.itemId);
          if (!pm.isNull()) indexItem->setData(Qt::DecorationRole, pm.scaled(24, 24));
          QString toolTip = QString("%1 (%2)")
                                .arg(QString::fromStdString(
                                    gameData_.getItemName(info.itemId)))
                                .arg(info.itemId);
          indexItem->setToolTip(toolTip);
        }
      }
      ui_->qValuesTable->setItem(i, 0, indexItem);
      QProgressBar *bar = new QProgressBar;
      bar->setRange(0, 100);
      ui_->qValuesTable->setCellWidget(i, 1, bar);
      ui_->qValuesTable->setRowHeight(i, 24);
      qValueBars_.append(bar);
    }
  } else {
    for (int i = 0; i < qValues.size(); ++i) {
      QTableWidgetItem *item = ui_->qValuesTable->item(i, 0);
      if (!item) {
        item = new QTableWidgetItem;
        ui_->qValuesTable->setItem(i, 0, item);
      }
      if (i < static_cast<int>(std::size(kActionInfos))) {
        const ActionInfo &info = kActionInfos[i];
        item->setData(Qt::DecorationRole, QVariant());
        item->setText(QString());
        if (info.type == ActionType::Text) {
          item->setText(QString::fromLatin1(info.text));
        } else if (info.type == ActionType::Skill) {
          QPixmap pm = getIconForSkillId(info.skillId);
          if (!pm.isNull()) item->setData(Qt::DecorationRole, pm.scaled(24,24));
          QString toolTip = QString("%1 (%2)")
                                .arg(QString::fromStdString(gameData_.getSkillName(info.skillId)))
                                .arg(info.skillId);
          item->setToolTip(toolTip);
        } else if (info.type == ActionType::Item) {
          QPixmap pm = getIconForItemId(info.itemId);
          if (!pm.isNull()) item->setData(Qt::DecorationRole, pm.scaled(24, 24));
          QString toolTip = QString("%1 (%2)")
                                .arg(QString::fromStdString(
                                    gameData_.getItemName(info.itemId)))
                                .arg(info.itemId);
          item->setToolTip(toolTip);
        }
      }
      ui_->qValuesTable->setRowHeight(i, 24);
    }
  }

  float minVal = std::numeric_limits<float>::infinity();
  float maxVal = -std::numeric_limits<float>::infinity();
  for (float v : qValues) {
    if (v == -std::numeric_limits<float>::infinity()) {
      continue;
    }
    if (v < minVal) {
      minVal = v;
    }
    if (v > maxVal) {
      maxVal = v;
    }
  }
  if (minVal == std::numeric_limits<float>::infinity()) {
    minVal = 0.0f;
    maxVal = 0.0f;
  }

  const float range = maxVal - minVal;
  for (int i = 0; i < qValues.size(); ++i) {
    QProgressBar *bar = qValueBars_.value(i);
    if (!bar) {
      continue;
    }
    if (qValues[i] == -std::numeric_limits<float>::infinity()) {
      bar->setValue(0);
      bar->setFormat("-inf");
    } else if (range == 0.0f) {
      bar->setValue(100);
      bar->setFormat(QString::number(qValues[i], 'f', 2));
    } else {
      float pctF = (qValues[i] - minVal) / range;
      int pct = static_cast<int>(pctF * 100.0f);
      if (pct < 0) pct = 0;
      if (pct == 0 && qValues[i] != minVal) pct = 1;
      if (pct > 100) pct = 100;
      bar->setValue(pct);
      bar->setFormat(QString::number(qValues[i], 'f', 2));
    }
  }
}

void CharacterDetailDialog::updateCharacterData(const CharacterData &data) {
  updateHpMp(data.currentHp, data.maxHp, data.currentMp, data.maxMp);
  updateItemCount(static_cast<sro::scalar_types::ReferenceObjectId>(5),
                  data.itemCounts.value(5));
  updateItemCount(static_cast<sro::scalar_types::ReferenceObjectId>(12),
                  data.itemCounts.value(12));
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

void CharacterDetailDialog::onItemCountUpdated(QString name,
                                               sro::scalar_types::ReferenceObjectId itemRefId,
                                               int count) {
  if (name == name_) {
    updateItemCount(itemRefId, count);
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

QPixmap CharacterDetailDialog::getIconForItemId(
    sro::scalar_types::ReferenceObjectId itemId) {
  if (itemIconCache_.contains(itemId)) {
    return itemIconCache_.value(itemId);
  }
  const gli::texture2d *texture = gameData_.getItemIcon(itemId);
  if (texture == nullptr) {
    return QPixmap();
  }
  try {
    QImage img = texture_to_image::texture2dToQImage(*texture);
    QPixmap pixmap = QPixmap::fromImage(img);
    itemIconCache_.insert(itemId, pixmap);
    return pixmap;
  } catch (const std::exception &ex) {
    LOG(WARNING) << "Failed to convert item icon for id " << itemId << ": "
                 << ex.what();
    return QPixmap();
  }
}
