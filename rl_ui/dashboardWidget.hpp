#ifndef DASHBOARD_WIDGET_HPP_
#define DASHBOARD_WIDGET_HPP_

#include <QStringList>
#include <QWidget>
#include <QMap>

#include "characterDetailDialog.hpp"

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

signals:
  void characterDataUpdated(QString name, CharacterData data);

private:
  Ui::DashboardWidget *ui;
  QMap<QString, CharacterData> characterData_;
  int ensureRowForCharacter(const QString &name);
  void showCharacterDetail(int row, int column);
};

#endif // DASHBOARD_WIDGET_HPP_
