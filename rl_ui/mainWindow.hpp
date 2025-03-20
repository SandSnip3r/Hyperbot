#ifndef MAINWINDOW_HPP_
#define MAINWINDOW_HPP_

#include "hyperbot.hpp"

#include <QMainWindow>
#include <QStringList>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(Hyperbot &hyperbot, QWidget *parent = nullptr);
  ~MainWindow();

public slots:
  void timerTriggered();
  void checkpointListReceived(QStringList list);

private:
  Ui::MainWindow *ui;
  Hyperbot &hyperbot_;
  QLineSeries *series_;
  QValueAxis *xAxis_;
  QTimer *timer_;

  void connectSignals();
  void testChart();
};
#endif // MAINWINDOW_HPP_
