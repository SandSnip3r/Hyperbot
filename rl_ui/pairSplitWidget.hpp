#ifndef PAIR_SPLIT_WIDGET_HPP_
#define PAIR_SPLIT_WIDGET_HPP_

#include <QWidget>

namespace Ui {
class PairSplitWidget;
}

class PairSplitWidget : public QWidget {
  Q_OBJECT

public:
  explicit PairSplitWidget(QWidget *parent = nullptr);
  ~PairSplitWidget();

private slots:
  void onTeamASelectionChanged(int row, int column);
  void onTeamBSelectionChanged(int row, int column);

private:
  Ui::PairSplitWidget *ui_;
  void updateDetailArea(const QString &teamAName, const QString &teamBName);
};

#endif // PAIR_SPLIT_WIDGET_HPP_
