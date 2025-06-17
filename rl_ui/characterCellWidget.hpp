#ifndef CHARACTER_CELL_WIDGET_HPP_
#define CHARACTER_CELL_WIDGET_HPP_

#include "characterDetailDialog.hpp"

#include <QWidget>

class QLabel;
class QProgressBar;

class CharacterCellWidget : public QWidget {
  Q_OBJECT
public:
  explicit CharacterCellWidget(QWidget *parent = nullptr);

  void setCharacterName(const QString &name);
  void updateCharacterData(const CharacterData &data);

signals:
  void expandRequested(CharacterCellWidget *cell);

protected:
  void mousePressEvent(QMouseEvent *event) override;

public slots:
  void toggleExpanded();

private:
  QLabel *nameLabel_;
  QProgressBar *hpBar_;
  QProgressBar *mpBar_;
  QLabel *stateLabel_;
  QWidget *detailWidget_;
  bool expanded_{false};
};

#endif // CHARACTER_CELL_WIDGET_HPP_
