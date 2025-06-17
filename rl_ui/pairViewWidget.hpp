#ifndef PAIR_VIEW_WIDGET_HPP_
#define PAIR_VIEW_WIDGET_HPP_

#include <QWidget>
#include <QStringList>

QT_BEGIN_NAMESPACE
namespace Ui {
class PairViewWidget;
}
QT_END_NAMESPACE

class PairViewWidget : public QWidget {
  Q_OBJECT
public:
  explicit PairViewWidget(QWidget *parent = nullptr);
  ~PairViewWidget();

  void setPairList(const QStringList &teamA, const QStringList &teamB);

private slots:
  void onPairSelected(int row);

private:
  Ui::PairViewWidget *ui_;
  QStringList teamA_;
  QStringList teamB_;
};

#endif // PAIR_VIEW_WIDGET_HPP_
