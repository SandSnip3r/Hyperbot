#ifndef DASHBOARD_WIDGET_HPP_
#define DASHBOARD_WIDGET_HPP_

#include <QStringList>
#include <QWidget>

namespace Ui {
class DashboardWidget;
}

class DashboardWidget : public QWidget {
  Q_OBJECT
public:
  explicit DashboardWidget(QWidget *parent = nullptr);
  ~DashboardWidget();

public slots:
  void onCharacterStatusReceived(QString name, int currentHp, int maxHp,
                                 int currentMp, int maxMp);
  void onActiveStateMachine(QString name, QString stateMachine);
  void clearStatusTable();

private:
  Ui::DashboardWidget *ui;
  int ensureRowForCharacter(const QString &name);
};

#endif // DASHBOARD_WIDGET_HPP_
