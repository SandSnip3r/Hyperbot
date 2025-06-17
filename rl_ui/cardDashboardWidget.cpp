#include "cardDashboardWidget.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QRegularExpression>

CardDashboardWidget::CardDashboardWidget(const sro::pk2::GameData &gameData,
                                         QWidget *parent)
    : QWidget(parent),
      filterEdit_(new QLineEdit(this)),
      scrollArea_(new QScrollArea(this)),
      gridWidget_(new QWidget(scrollArea_)),
      gridLayout_(new QGridLayout(gridWidget_)),
      gameData_(gameData) {
  QVBoxLayout *mainLayout = new QVBoxLayout(this);
  QHBoxLayout *filterLayout = new QHBoxLayout;
  QLabel *label = new QLabel(tr("Filter:"), this);
  filterLayout->addWidget(label);
  filterLayout->addWidget(filterEdit_);
  mainLayout->addLayout(filterLayout);
  scrollArea_->setWidgetResizable(true);
  scrollArea_->setWidget(gridWidget_);
  mainLayout->addWidget(scrollArea_);
  gridLayout_->setContentsMargins(2, 2, 2, 2);
  gridLayout_->setSpacing(4);
  connect(filterEdit_, &QLineEdit::textChanged, this,
          &CardDashboardWidget::filterTextChanged);
}

CardDashboardWidget::~CardDashboardWidget() {}

CharacterCardWidget *CardDashboardWidget::ensureCard(const QString &name) {
  if (!cardWidgets_.contains(name)) {
    int index = cardWidgets_.size();
    int row = index / 4;
    int col = index % 4;
    CharacterCardWidget *card = new CharacterCardWidget(gridWidget_);
    card->setCharacterName(name);
    card->setPairColor(colorForPair(name));
    gridLayout_->addWidget(card, row, col);
    cardWidgets_.insert(name, card);
    connect(card, &CharacterCardWidget::clicked, this,
            &CardDashboardWidget::cardSelected);
  }
  return cardWidgets_.value(name);
}

QColor CardDashboardWidget::colorForPair(const QString &name) const {
  QRegularExpression re("RL_(\\d+)");
  QRegularExpressionMatch match = re.match(name);
  int id = 0;
  if (match.hasMatch()) {
    id = match.captured(1).toInt();
  } else {
    id = name.toInt();
  }
  int hue = (id * 45) % 360;
  return QColor::fromHsv(hue, 160, 200);
}

void CardDashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                                    int maxHp, int currentMp,
                                                    int maxMp) {
  CharacterCardWidget *card = ensureCard(name);
  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;
  card->updateCharacterData(data);
  emit characterDataUpdated(name, data);
  applyFilter();
}

void CardDashboardWidget::onActiveStateMachine(QString name,
                                               QString stateMachine) {
  CharacterCardWidget *card = ensureCard(name);
  characterData_[name].stateMachine = stateMachine;
  card->updateCharacterData(characterData_[name]);
  emit characterDataUpdated(name, characterData_.value(name));
  applyFilter();
}

void CardDashboardWidget::onSkillCooldowns(QString name,
                                           QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  emit characterDataUpdated(name, characterData_.value(name));
}

void CardDashboardWidget::clearCards() {
  qDeleteAll(cardWidgets_);
  cardWidgets_.clear();
  characterData_.clear();
}

void CardDashboardWidget::onHyperbotConnected() { clearCards(); }

void CardDashboardWidget::filterTextChanged(const QString &text) {
  Q_UNUSED(text);
  applyFilter();
}

void CardDashboardWidget::applyFilter() {
  const QString filter = filterEdit_->text().trimmed();
  for (auto it = cardWidgets_.begin(); it != cardWidgets_.end(); ++it) {
    const QString name = it.key();
    CharacterCardWidget *card = it.value();
    const CharacterData &data = characterData_.value(name);
    bool visible = filter.isEmpty() ||
                   name.contains(filter, Qt::CaseInsensitive) ||
                   data.stateMachine.contains(filter, Qt::CaseInsensitive);
    card->setVisible(visible);
  }
}
