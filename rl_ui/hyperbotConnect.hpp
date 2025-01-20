#ifndef HYPERBOTCONNECT_HPP_
#define HYPERBOTCONNECT_HPP_

#include "config.hpp"
#include "hyperbot.hpp"

#include <QMainWindow>

namespace Ui {
class HyperbotConnect;
}

class HyperbotConnect : public QMainWindow {
  Q_OBJECT

public:
  explicit HyperbotConnect(Config &&config, Hyperbot &hyperbot, QWidget *parent = nullptr);
  ~HyperbotConnect();

signals:
private slots:
  void connectClicked();
private:
  Ui::HyperbotConnect *ui;
  Config config_;
  Hyperbot &hyperbot_;
};

#endif // HYPERBOTCONNECT_HPP_
