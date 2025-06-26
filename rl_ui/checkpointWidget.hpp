#ifndef CHECKPOINT_WIDGET_HPP_
#define CHECKPOINT_WIDGET_HPP_

#include "hyperbot.hpp"

#include <QStringList>
#include <QStandardItemModel>
#include <QWidget>

namespace Ui {
class CheckpointWidget;
}

class CheckpointWidget : public QWidget {
  Q_OBJECT

public:
  explicit CheckpointWidget(QWidget *parent = nullptr);
  ~CheckpointWidget();

  void setHyperbot(Hyperbot &hyperbot);
public slots:
  void onCheckpointListReceived(QList<CheckpointInfo> checkpointList);
  void onSaveCheckpointClicked();
  void onDeleteCheckpointClicked();
  void onCheckpointLoaded(QString checkpointName);

private:
  Ui::CheckpointWidget *ui;
  QStandardItemModel *checkpointModel_{nullptr};
  Hyperbot *hyperbot_{nullptr};
};

#endif // CHECKPOINT_WIDGET_HPP_
