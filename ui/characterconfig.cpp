#include "characterconfig.h"
#include "ui_characterconfig.h"

#include "ui_proto/position.pb.h"

#include <QString>

#include <iostream>

CharacterConfig::CharacterConfig(QWidget *parent) : QGroupBox(parent), ui(new Ui::CharacterConfig) {
  ui->setupUi(this);
}

CharacterConfig::~CharacterConfig() {
  delete ui;
}

void CharacterConfig::setEventHandlerAndRequester(EventHandler *eventHandler, Requester *requester) {
  eventHandler_ = eventHandler;
  requester_ = requester;

  // connect(eventHandler_, &EventHandler::configReceived, this, &CharacterConfig::configReceived);
  connect(ui->saveButton, &QPushButton::clicked, this, &CharacterConfig::saveAndSendConfig);
  connect(ui->setCurrentAsTrainingCenterButton, &QPushButton::clicked, requester_, &Requester::setCurrentPositionAsTrainingCenter);
}

void CharacterConfig::saveAndSendConfig() {
  if (!config_) {
    std::cout << "We havent yet received a config; it would be unwise to overwrite the one the bot has" << std::endl;
    return;
  }
  // We'll overwrite config_
  updateConfigFromUi();
  requester_->sendConfig(*config_);
}

void CharacterConfig::configReceived(proto::old_config::Config config) {
  // Save the config.
  config_ = config;

  const proto::character_config::CharacterConfig *characterConfig = getCurrentCharacterConfig();
  if (characterConfig == nullptr) {
    return;
  }
  populateUiFromConfig(*characterConfig);
}

const proto::character_config::CharacterConfig* CharacterConfig::getCurrentCharacterConfig() const {
  if (!config_) {
    // Don't have a config
    throw std::runtime_error("Cannot get character config with no config");
  }

  if (!config_->has_character_to_login()) {
    std::cout << "Config has no character to login" << std::endl;
    return nullptr;
  }

  const std::string &character = config_->character_to_login();
  for (int i=0; i<config_->character_configs_size(); ++i) {
    const proto::character_config::CharacterConfig &characterConfig = config_->character_configs(i);
    if (characterConfig.character_name() == character) {
      return &characterConfig;
    }
  }
}

proto::character_config::CharacterConfig* CharacterConfig::getMutableCurrentCharacterConfig() {
  if (!config_) {
    // Don't have a config
    throw std::runtime_error("Cannot get character config with no config");
  }

  if (!config_->has_character_to_login()) {
    std::cout << "Config has no character to login" << std::endl;
    return nullptr;
  }

  const std::string &character = config_->character_to_login();
  for (int i=0; i<config_->character_configs_size(); ++i) {
    proto::character_config::CharacterConfig *characterConfig = config_->mutable_character_configs(i);
    if (characterConfig->character_name() == character) {
      return characterConfig;
    }
  }
}

void CharacterConfig::populateUiFromConfig(const proto::character_config::CharacterConfig &config) {
  ui->characterNameLineEdit->setText(QString::fromStdString(config.character_name()));
  ui->usernameLineEdit->setText(QString::fromStdString(config.username()));
  ui->passwordLineEdit->setText(QString::fromStdString(config.password()));
  const proto::character_config::AutopotionConfig &autoPotionConfig = config.autopotion_config();
  ui->hpThresholdSpinBox->setValue(100 * autoPotionConfig.hp_threshold()/1.0);
  ui->mpThresholdSpinBox->setValue(100 * autoPotionConfig.mp_threshold()/1.0);
  ui->vigorHpThresholdSpinBox->setValue(100 * autoPotionConfig.vigor_hp_threshold()/1.0);
  ui->vigorMpThresholdSpinBox->setValue(100 * autoPotionConfig.vigor_mp_threshold()/1.0);
  const proto::character_config::TrainingConfig &trainingConfig = config.training_config();
  ui->trainingCenterRegionLineEdit->setText(QString::number(trainingConfig.center().regionid()));
  ui->trainingCenterXLineEdit->setText(QString::number(trainingConfig.center().x()));
  ui->trainingCenterYLineEdit->setText(QString::number(trainingConfig.center().y()));
  ui->trainingCenterZLineEdit->setText(QString::number(trainingConfig.center().z()));
  ui->trainingRadiusLineEdit->setText(QString::number(trainingConfig.radius()));
}

void CharacterConfig::updateConfigFromUi() {
  proto::character_config::CharacterConfig *characterConfig = getMutableCurrentCharacterConfig();
  if (characterConfig == nullptr) {
    std::cout << "No character config to write to" << std::endl;
    return;
  }
  updateAutopotionConfigFromUi(*characterConfig->mutable_autopotion_config());
  updateTrainingConfigFromUi(*characterConfig->mutable_training_config());
}

void CharacterConfig::updateAutopotionConfigFromUi(proto::character_config::AutopotionConfig &autopotionConfig) {
  autopotionConfig.set_hp_threshold(static_cast<double>(ui->hpThresholdSpinBox->value())/ui->hpThresholdSpinBox->maximum());
  autopotionConfig.set_mp_threshold(static_cast<double>(ui->mpThresholdSpinBox->value())/ui->mpThresholdSpinBox->maximum());
  autopotionConfig.set_vigor_hp_threshold(static_cast<double>(ui->vigorHpThresholdSpinBox->value())/ui->vigorHpThresholdSpinBox->maximum());
  autopotionConfig.set_vigor_mp_threshold(static_cast<double>(ui->vigorMpThresholdSpinBox->value())/ui->vigorMpThresholdSpinBox->maximum());
}

void CharacterConfig::updateTrainingConfigFromUi(proto::character_config::TrainingConfig &trainingConfig) {
  proto::position::Position &centerPosition = *trainingConfig.mutable_center();
  centerPosition.set_regionid(ui->trainingCenterRegionLineEdit->text().toInt());
  centerPosition.set_x(ui->trainingCenterXLineEdit->text().toDouble());
  centerPosition.set_y(ui->trainingCenterYLineEdit->text().toDouble());
  centerPosition.set_z(ui->trainingCenterZLineEdit->text().toDouble());
  trainingConfig.set_radius(ui->trainingRadiusLineEdit->text().toDouble());
}