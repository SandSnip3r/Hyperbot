#include "checkpointWidget.hpp"
#include "ui_checkpointwidget.h"

#include <QMessageBox>
#include <QStringListModel>

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

CheckpointWidget::CheckpointWidget(QWidget *parent) : QWidget(parent), ui(new Ui::CheckpointWidget) {
  ui->setupUi(this);
  ui->loadCheckpointButton->setEnabled(false);

  // Create an empty model and set it on the view
  checkpointModel_ = new QStringListModel(this);
  ui->checkpointsListView->setModel(checkpointModel_);

  // Get the selection model for the list view
  QItemSelectionModel *selectionModel = ui->checkpointsListView->selectionModel();

  // Connect to the selectionChanged signal to update the button state
  connect(selectionModel, &QItemSelectionModel::selectionChanged, this, [this, selectionModel]() {
    // Check the total number of selected indexes in the list view.
    int selectedCount = selectionModel->selectedIndexes().size();
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

  // Connect the button click to load the selected checkpoint
  connect(ui->loadCheckpointButton, &QPushButton::clicked, this, [this]() {
    // Ensure exactly one item is selected before proceeding
    QItemSelectionModel *selectionModel = ui->checkpointsListView->selectionModel();
    QModelIndexList selected = selectionModel->selectedIndexes();
    if (selected.size() == 1) {
      QString selectedCheckpoint = selected.first().data(Qt::DisplayRole).toString();
      hyperbot_->loadCheckpoint(selectedCheckpoint);
    }
  });
}

void CheckpointWidget::onCheckpointListReceived(QStringList list) {
  VLOG(1) << "Received checkpoint list: " << list.join(", ").toStdString();
  // Update the model's data instead of creating a new one
  checkpointModel_->setStringList(list);
  ui->saveCheckpointButton->setEnabled(true);
}

void CheckpointWidget::onSaveCheckpointClicked() {
  QString checkpointName = ui->checkpointNameLineEdit->text();
  hyperbot_->saveCheckpoint(checkpointName);
}

void CheckpointWidget::onDeleteCheckpointClicked() {
  QModelIndexList selectedIndices = ui->checkpointsListView->selectionModel()->selectedIndexes();
  std::string confirmationString = absl::StrFormat("Are you sure you want to delete checkpoint(s):\n%s", absl::StrJoin(selectedIndices, "\n", [](std::string *out, QModelIndex index) {
    absl::StrAppend(out, index.data(Qt::DisplayRole).toString().toStdString());
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
    checkpointNamesToDelete.append(index.data(Qt::DisplayRole).toString());
  }
  hyperbot_->deleteCheckpoints(checkpointNamesToDelete);
}
