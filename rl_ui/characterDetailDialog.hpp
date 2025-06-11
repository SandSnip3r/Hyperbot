#ifndef CHARACTER_DETAIL_DIALOG_HPP_
#define CHARACTER_DETAIL_DIALOG_HPP_

#include <QDialog>

namespace Ui {
class CharacterDetailDialog;
}

struct CharacterData {
  int currentHp{0};
  int maxHp{0};
  int currentMp{0};
  int maxMp{0};
  QString stateMachine;
};

class CharacterDetailDialog : public QDialog {
  Q_OBJECT
public:
  explicit CharacterDetailDialog(QWidget *parent = nullptr);
  ~CharacterDetailDialog();

  void setCharacterName(const QString &name);
  void setCharacterData(const CharacterData &data);

private:
  Ui::CharacterDetailDialog *ui;
};

#endif // CHARACTER_DETAIL_DIALOG_HPP_
