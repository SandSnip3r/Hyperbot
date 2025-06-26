#include "checkpointWidget.hpp"
#include "ui_checkpointwidget.h"

#include <QMessageBox>
#include <QStandardItemModel>
#include <QDateTime>
#include <QHeaderView>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

CheckpointWidget::CheckpointWidget(QWidget *parent) : QWidget(parent), ui(new Ui::CheckpointWidget) {
  ui->setupUi(this);
  ui->loadCheckpointButton->setEnabled(false);

  // Create an empty model and set it on the view
  checkpointModel_ = new QStandardItemModel(this);
  checkpointModel_->setHorizontalHeaderLabels({"Name", "Save Date", "Train Step Count"});
  ui->checkpointsTableView->setModel(checkpointModel_);
  ui->checkpointsTableView->setSelectionBehavior(QAbstractItemView::SelectRows);
  ui->checkpointsTableView->horizontalHeader()->setStretchLastSection(true);

  // Get the selection model for the list view
  QItemSelectionModel *selectionModel = ui->checkpointsTableView->selectionModel();

  // Connect to the selectionChanged signal to update the button state
  connect(selectionModel, &QItemSelectionModel::selectionChanged, this, [this, selectionModel]() {
    // Check the total number of selected indexes in the list view.
    int selectedCount = selectionModel->selectedRows().size();
    ui->loadCheckpointButton->setEnabled(selectedCount == 1);
    ui->deleteCheckpointButton->setEnabled(selectedCount > 0);
  });

  connect(ui->saveCheckpointButton, &QPushButton::clicked, this, &CheckpointWidget::onSaveCheckpointClicked);
  connect(ui->deleteCheckpointButton, &QPushButton::clicked, this, &CheckpointWidget::onDeleteCheckpointClicked);
}

CheckpointWidget::~CheckpointWidget() {
  delete ui;
}

void CheckpointWidget::setHyperbot(Hyperbot &hyperbot) {
  hyperbot_ = &hyperbot;
  connect(hyperbot_, &Hyperbot::checkpointListReceived, this, &CheckpointWidget::onCheckpointListReceived);
  connect(hyperbot_, &Hyperbot::checkpointAlreadyExists, this, [this](QString checkpointName) {
    QMessageBox::information(this, "Checkpoint already exists", "Checkpoint \"" + checkpointName + "\" already exists.");
  });
  connect(hyperbot_, &Hyperbot::savingCheckpoint, this, [this](){
    ui->saveCheckpointButton->setEnabled(false);
  });
  connect(hyperbot_, &Hyperbot::checkpointLoaded, this, &CheckpointWidget::onCheckpointLoaded);

  // Connect the button click to load the selected checkpoint
  connect(ui->loadCheckpointButton, &QPushButton::clicked, this, [this]() {
    // Ensure exactly one item is selected before proceeding
    QItemSelectionModel *selectionModel = ui->checkpointsTableView->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();
    if (selected.size() == 1) {
      const int row = selected.first().row();
      QString selectedCheckpoint = checkpointModel_->item(row, 0)->text();
      hyperbot_->loadCheckpoint(selectedCheckpoint);
    }
  });
}

void CheckpointWidget::onCheckpointListReceived(QList<CheckpointInfo> list) {
  checkpointModel_->removeRows(0, checkpointModel_->rowCount());
  int row = 0;
  for (const CheckpointInfo &info : list) {
    QList<QStandardItem *> items;
    items.append(new QStandardItem(info.name));
    QDateTime dt = QDateTime::fromMSecsSinceEpoch(info.timestampMs);
    items.append(new QStandardItem(dt.toString()));
    items.append(new QStandardItem(QString::number(info.trainStepCount)));
    checkpointModel_->appendRow(items);
    ++row;
  }
  ui->saveCheckpointButton->setEnabled(true);
}

void CheckpointWidget::onSaveCheckpointClicked() {
  QString checkpointName = ui->checkpointNameLineEdit->text();
  hyperbot_->saveCheckpoint(checkpointName);
}

void CheckpointWidget::onDeleteCheckpointClicked() {
  QModelIndexList selectedIndices = ui->checkpointsTableView->selectionModel()->selectedRows();
  std::string confirmationString = absl::StrFormat("Are you sure you want to delete checkpoint(s):\n%s", absl::StrJoin(selectedIndices, "\n", [](std::string *out, QModelIndex index) {
    absl::StrAppend(out, checkpointModel_->item(index.row(), 0)->text().toStdString());
  }));
  QMessageBox::StandardButton reply;
  reply = QMessageBox::question(
              this,
              "Confirm Delete",
              QString::fromStdString(confirmationString),
              QMessageBox::Yes | QMessageBox::No
          );

  if (reply != QMessageBox::Yes) {
    return;
  }

  // User clicked "Yes".
  QStringList checkpointNamesToDelete;
  for (const QModelIndex &index : selectedIndices) {
    checkpointNamesToDelete.append(checkpointModel_->item(index.row(), 0)->text());
  }
  hyperbot_->deleteCheckpoints(checkpointNamesToDelete);
}

void CheckpointWidget::onCheckpointLoaded(QString checkpointName) {
  ui->currentlyLoadedCheckpoint->setText(checkpointName);
}
