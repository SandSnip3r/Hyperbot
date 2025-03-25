#ifndef CHECKPOINT_WIDGET_HPP_
#define CHECKPOINT_WIDGET_HPP_

#include "hyperbot.hpp"

#include <QStringList>
#include <QStringListModel>
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
  void onCheckpointListReceived(QStringList checkpointList);
  void onSaveCheckpointClicked();

private:
  Ui::CheckpointWidget *ui;
  QStringListModel *checkpointModel_{nullptr};
  Hyperbot *hyperbot_{nullptr};
};

#endif // CHECKPOINT_WIDGET_HPP_
