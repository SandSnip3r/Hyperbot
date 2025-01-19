#ifndef MAINWINDOW_HPP_
#define MAINWINDOW_HPP_

#include "hyperbot.hpp"

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();
  void setBot(Hyperbot &&bot);

private:
  Ui::MainWindow *ui;
  Hyperbot hyperbot_;
};
#endif // MAINWINDOW_HPP_
