#include "checkpointWidget.hpp"
#include "ui_checkpointwidget.h"

#include <QMessageBox>
#include <QStringListModel>

#include <absl/log/log.h>

CheckpointWidget::CheckpointWidget(QWidget *parent) : QWidget(parent), ui(new Ui::CheckpointWidget) {
  ui->setupUi(this);
}

CheckpointWidget::~CheckpointWidget() {
  delete ui;
}

void CheckpointWidget::setHyperbot(Hyperbot &hyperbot) {
  hyperbot_ = &hyperbot;
  connect(hyperbot_, &Hyperbot::checkpointListReceived, this, &CheckpointWidget::onCheckpointListReceived);
  connect(hyperbot_, &Hyperbot::checkpointAlreadyExists, [this](QString checkpointName) {
    QMessageBox::information(this, "Checkpoint already exists", "Checkpoint \"" + checkpointName + "\" already exists.");
  });
  connect(hyperbot_, &Hyperbot::savingCheckpoint, [this](){
    ui->saveCheckpointButton->setEnabled(false);
  });
  connect(ui->saveCheckpointButton, &QPushButton::clicked, this, &CheckpointWidget::onSaveCheckpointClicked);
}

void CheckpointWidget::onCheckpointListReceived(QStringList list) {
  VLOG(1) << "Received checkpoint list: " << list.join(", ").toStdString();
  QStringListModel *checkpointModel = new QStringListModel(list);
  ui->checkpointsListView->setModel(checkpointModel);
  ui->saveCheckpointButton->setEnabled(true);
}

void CheckpointWidget::onSaveCheckpointClicked() {
  QString checkpointName = ui->checkpointNameLineEdit->text();
  hyperbot_->saveCheckpoint(checkpointName);
}
