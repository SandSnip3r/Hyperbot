#include "checkpointWidget.hpp"
#include "ui_checkpointwidget.h"

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
}

void CheckpointWidget::onCheckpointListReceived(QStringList list) {
  VLOG(1) << "Received checkpoint list: " << list.join(", ").toStdString();
  QStringListModel *checkpointModel = new QStringListModel(list);
  ui->checkpointsListView->setModel(checkpointModel);
}
