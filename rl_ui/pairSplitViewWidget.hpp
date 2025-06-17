#ifndef PAIR_SPLIT_VIEW_WIDGET_HPP_
#define PAIR_SPLIT_VIEW_WIDGET_HPP_

#include <QWidget>

namespace Ui {
class PairSplitViewWidget;
}

class PairSplitViewWidget : public QWidget {
  Q_OBJECT
public:
  explicit PairSplitViewWidget(QWidget *parent = nullptr);
  ~PairSplitViewWidget();

private:
  Ui::PairSplitViewWidget *ui;
};

#endif // PAIR_SPLIT_VIEW_WIDGET_HPP_
