#include "bot.hpp"
#include "logging.hpp"

#include "packet/building/clientAgentActionDeselectRequest.hpp"
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentActionTalkRequest.hpp"
#include "packet/building/clientAgentAuthRequest.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "packet/building/clientAgentCharacterSelectionJoinRequest.hpp"
#include "packet/building/clientAgentInventoryOperationRequest.hpp"
#include "packet/building/clientAgentInventoryStorageOpenRequest.hpp"
#include "packet/building/clientGatewayLoginIbuvAnswer.hpp"
#include "packet/building/clientGatewayLoginRequest.hpp"
// TODO: <remove>
// For quicker development, when we spawn in, set ourself as visible and put on a PVP cape
#include "packet/building/clientAgentFreePvpUpdateRequest.hpp"
#include "packet/building/clientAgentOperatorRequest.hpp"
// </remove>
#include "type_id/categories.hpp"

#include "pathfinder.h"

#include <silkroad_lib/game_constants.h>
#include <silkroad_lib/position_math.h>

Bot::Bot(const config::Config &config,
         const pk2::GameData &gameData,
         Proxy &proxy,
         broker::PacketBroker &packetBroker,
         broker::EventBroker &eventBroker) :
      config_(config),
      gameData_(gameData),
      proxy_(proxy),
      packetBroker_(packetBroker),
      eventBroker_(eventBroker) {
  // Check that the config has all the data we need
  std::unique_lock<std::mutex> configLock(config_.mutex());
  const auto &configProto = config_.configProto();
  if (!configProto.has_character_to_login()) {
    throw std::runtime_error("Config does not contain a character to login");
  }
  const std::string characterToLogin = configProto.character_to_login();
  if (!config_.haveCharacterConfig(characterToLogin)) {
    throw std::runtime_error("Config does not contain character config for character to login");
  }

  std::ifstream estVisRangeFile{kEstVisRangeFilename};
  if (estVisRangeFile) {
    estVisRangeFile >> worldState_.selfState().estimatedVisibilityRange;
    LOG() << "Parsed estimated visibility range from file as " << worldState_.selfState().estimatedVisibilityRange << std::endl;
  }

  autoPotionStateMachine_ = std::make_unique<state::machine::AutoPotion>(*this);
}

void Bot::initialize() {
  subscribeToEvents();
  packetProcessor_.initialize();
}

const config::Config& Bot::config() const {
  return config_;
}

const proto::config::CharacterConfig& Bot::currentCharacterConfig() const {
  if (!worldState_.selfState().spawned()) {
    throw std::runtime_error("Not yet spawned, don't know which character config to fetch");
  }
  return config_.getCharacterConfig(worldState_.selfState().name);
}

const pk2::GameData& Bot::gameData() const {
  return gameData_;
}

Proxy& Bot::proxy() const {
  return proxy_;
}

broker::PacketBroker& Bot::packetBroker() const {
  return packetBroker_;
}

broker::EventBroker& Bot::eventBroker() {
  return eventBroker_;
}

const state::WorldState& Bot::worldState() const {
  return worldState_;
}

state::EntityTracker& Bot::entityTracker() {
  return worldState_.entityTracker();
}

state::Self& Bot::selfState() {
  return worldState_.selfState();
}

const state::Self& Bot::selfState() const {
  return worldState_.selfState();
}

void Bot::subscribeToEvents() {
  auto eventHandleFunction = std::bind(&Bot::handleEvent, this, std::placeholders::_1);
  // Bot actions from UI
  eventBroker_.subscribeToEvent(event::EventCode::kStartTraining, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStopTraining, eventHandleFunction);
  // Debug help
  eventBroker_.subscribeToEvent(event::EventCode::kInjectPacket, eventHandleFunction);
  // Login events
  eventBroker_.subscribeToEvent(event::EventCode::kStateShardIdUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateConnectedToAgentServerUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateReceivedCaptchaPromptUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kLoggedIn, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStateCharacterListUpdated, eventHandleFunction);
  // Movement events
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementBegan, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityMovementTimerEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityEnteredGeometry, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityExitedGeometry, eventHandleFunction);
  // Character info events
  eventBroker_.subscribeToEvent(event::EventCode::kSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemUseFailed, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityHpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMaxHpMpChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStatesChanged, eventHandleFunction);

  // Misc
  eventBroker_.subscribeToEvent(event::EventCode::kEntityDeselected, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntitySelected, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kNpcTalkStart, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageInitialized, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kStorageUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kRepairSuccessful, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntitySpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityDespawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityBodyStateChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityLifeStateChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemUseTimeout, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillCastTimeout, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityOwnershipRemoved, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockedBack, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockedDown, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockbackStunEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKnockdownStunEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kMovementRequestTimedOut, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kItemCooldownEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryItemUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kHwanPointsUpdated, eventHandleFunction);

  // Skills
  eventBroker_.subscribeToEvent(event::EventCode::kSkillBegan, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillEnded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kOurSkillFailed, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kPlayerCharacterBuffAdded, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kPlayerCharacterBuffRemoved, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kOurCommandError, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kSkillCooldownEnded, eventHandleFunction);

  eventBroker_.subscribeToEvent(event::EventCode::kStateMachineActiveTooLong, eventHandleFunction);
}

void Bot::handleEvent(const event::Event *event) {
  std::unique_lock<std::mutex> selfStateLock(worldState_.selfState().selfMutex);

  try {
    const auto eventCode = event->eventCode;
    switch (eventCode) {
      // Bot actions from UI
      case event::EventCode::kStartTraining: {
        handleStartTraining();
        break;
      }
      case event::EventCode::kStopTraining: {
        handleStopTraining();
        break;
      }

      // Debug help
      case event::EventCode::kInjectPacket: {
        const event::InjectPacket &castedEvent = dynamic_cast<const event::InjectPacket&>(*event);
        handleInjectPacket(castedEvent);
        break;
      }

      // Login events
      case event::EventCode::kStateShardIdUpdated: {
        handleStateShardIdUpdated();
        break;
      }
      case event::EventCode::kStateConnectedToAgentServerUpdated: {
        handleStateConnectedToAgentServerUpdated();
        break;
      }
      case event::EventCode::kStateReceivedCaptchaPromptUpdated: {
        handleStateReceivedCaptchaPromptUpdated();
        break;
      }
      case event::EventCode::kLoggedIn: {
        handleLoggedIn();
        break;
      }
      case event::EventCode::kStateCharacterListUpdated: {
        handleStateCharacterListUpdated();
        break;
      }

      // Movement events
      case event::EventCode::kEntityMovementBegan: {
        const auto &castedEvent = dynamic_cast<const event::EntityMovementBegan&>(*event);
        handleEntityMovementBegan(castedEvent);
        break;
      }
      case event::EventCode::kEntityMovementEnded: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kEntityMovementTimerEnded: {
        const auto &castedEvent = dynamic_cast<const event::EntityMovementTimerEnded&>(*event);
        handleEntityMovementTimerEnded(castedEvent.globalId);
        break;
      }
      case event::EventCode::kEntityEnteredGeometry: {
        const auto &castedEvent = dynamic_cast<const event::EntityEnteredGeometry&>(*event);
        handleEntityEnteredGeometry(castedEvent);
        break;
      }
      case event::EventCode::kEntityExitedGeometry: {
        const auto &castedEvent = dynamic_cast<const event::EntityExitedGeometry&>(*event);
        handleEntityExitedGeometry(castedEvent);
        break;
      }

      // Character info events
      case event::EventCode::kSpawned: {
        handleSpawned(event);
        break;
      }
      case event::EventCode::kItemUseFailed: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kEntityHpChanged: {
        const event::EntityHpChanged &castedEvent = dynamic_cast<const event::EntityHpChanged&>(*event);
        if (castedEvent.globalId == worldState_.selfState().globalId) {
          handleVitalsChanged();
        }
        break;
      }
      case event::EventCode::kMpChanged:
      case event::EventCode::kMaxHpMpChanged: {
        handleVitalsChanged();
        break;
      }
      case event::EventCode::kStatesChanged: {
        handleStatesChanged();
        break;
      }

      // Misc
      case event::EventCode::kStorageInitialized:
      case event::EventCode::kEntityDeselected:
      case event::EventCode::kEntitySelected:
      case event::EventCode::kNpcTalkStart:
      case event::EventCode::kRepairSuccessful: {
        onUpdate();
        break;
      }
      case event::EventCode::kInventoryUpdated: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kStorageUpdated: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kEntitySpawned: {
        const auto &castedEvent = dynamic_cast<const event::EntitySpawned&>(*event);
        entitySpawned(castedEvent);
        break;
      }
      case event::EventCode::kEntityDespawned: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kEntityBodyStateChanged: {
        const auto &castedEvent = dynamic_cast<const event::EntityBodyStateChanged&>(*event);
        handleBodyStateChanged(castedEvent);
        break;
      }
      case event::EventCode::kEntityLifeStateChanged: {
        const auto &castedEvent = dynamic_cast<const event::EntityLifeStateChanged&>(*event);
        const auto &character = worldState_.getEntity<entity::Character>(castedEvent.globalId);
        if (character.globalId != selfState().globalId && character.lifeState == sro::entity::LifeState::kDead) {
          static int deathCount = 0;
          LOG() << ++deathCount << " kill(s)" << std::endl;
        }
        onUpdate(event);
        break;
      }
      case event::EventCode::kItemUseTimeout: {
        const auto &castedEvent = dynamic_cast<const event::ItemUseTimeout&>(*event);
        itemUseTimedOut(castedEvent);
        break;
      }
      case event::EventCode::kSkillCastTimeout: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kEntityOwnershipRemoved: {
        const auto &castedEvent = dynamic_cast<const event::EntityOwnershipRemoved&>(*event);
        onUpdate();
        break;
      }
      case event::EventCode::kKnockedBack: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kKnockedDown: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kKnockbackStunEnded: {
        handleKnockbackStunEnded();
        break;
      }
      case event::EventCode::kKnockdownStunEnded: {
        handleKnockdownStunEnded();
        break;
      }
      case event::EventCode::kMovementRequestTimedOut: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kItemCooldownEnded: {
        const auto &castedEvent = dynamic_cast<const event::ItemCooldownEnded&>(*event);
        handleItemCooldownEnded(castedEvent);
        break;
      }
      case event::EventCode::kInventoryItemUpdated: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kHwanPointsUpdated: {
        onUpdate(event);
        break;
      }

      // Skills
      case event::EventCode::kSkillBegan: {
        const auto &castedEvent = dynamic_cast<const event::SkillBegan&>(*event);
        handleSkillBegan(castedEvent);
        break;
      }
      case event::EventCode::kSkillEnded: {
        const auto &castedEvent = dynamic_cast<const event::SkillEnded&>(*event);
        handleSkillEnded(castedEvent);
        break;
      }
      case event::EventCode::kOurSkillFailed: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kPlayerCharacterBuffAdded: {
        break;
      }
      case event::EventCode::kPlayerCharacterBuffRemoved: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kOurCommandError: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kStateMachineActiveTooLong: {
        onUpdate(event);
        break;
      }
      case event::EventCode::kSkillCooldownEnded: {
        const auto &castedEvent = dynamic_cast<const event::SkillCooldownEnded&>(*event);
        handleSkillCooldownEnded(castedEvent);
        break;
      }
      default: {
        LOG() << "Unhandled event subscribed to. Code:" << static_cast<int>(eventCode) << '\n';
        break;
      }
    }
  } catch (std::exception &ex) {
    LOG() << "Error while handling event " << static_cast<int>(event->eventCode) << "!\n Error: \"" << ex.what() << '"' << std::endl;
  }
}

// ============================================================================================================================
// ====================================================Main Logic Game Loop====================================================
// ============================================================================================================================

void Bot::onUpdate(const event::Event *event) {
  // Highest priority is our vitals, we will try to heal even if we're not training
  if (autoPotionStateMachine_ && !autoPotionStateMachine_->done()) {
    autoPotionStateMachine_->onUpdate(event);
  }

  if (!worldState_.selfState().trainingIsActive) {
    // Not training, nothing else to do
    return;
  }

  if (bottingStateMachine_ && !bottingStateMachine_->done()) {
    bottingStateMachine_->onUpdate(event);
  }
}

// ============================================================================================================================
// =====================================================Bot actions from UI====================================================
// ============================================================================================================================

void Bot::handleStartTraining() {
  startTraining();
}

void Bot::handleStopTraining() {
  stopTraining();
}

void Bot::startTraining() {
  if (worldState_.selfState().trainingIsActive) {
    LOG() << "Asked to start training, but we're already training" << std::endl;
    return;
  }

  if (bottingStateMachine_) {
    throw std::runtime_error("Asked to start training, but already have a botting state machine");
  }

  LOG() << "Starting training" << std::endl;
  worldState_.selfState().trainingIsActive = true;
  // TODO: Should we stop whatever we're doing?
  //  For example, if we're running, stop where we are.

  // Initialize state machine
  bottingStateMachine_ = std::make_unique<state::machine::Botting>(*this);

  // Trigger onUpdate
  onUpdate();
}

void Bot::stopTraining() {
  if (worldState_.selfState().trainingIsActive) {
    // TODO: Need to cleanup current action to avoid leaving the client in a bad state
    //  Ex. Need to close a shop npc dialog
    LOG() << "Stopping training" << std::endl;
    worldState_.selfState().trainingIsActive = false;
    bottingStateMachine_.reset();
  } else {
    LOG() << "Asked to stop training, but we werent training" << std::endl;
  }
}

// ============================================================================================================================
// =========================================================Debug help=========================================================
// ============================================================================================================================

void Bot::handleInjectPacket(const event::InjectPacket &castedEvent) {
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
  PacketContainer::Direction direction = (castedEvent.direction == event::InjectPacket::Direction::kClientToServer) ? PacketContainer::Direction::kClientToServer : PacketContainer::Direction::kServerToClient;
  StreamUtility stream;
  for (const auto i : castedEvent.data) {
    stream.Write<uint8_t>(i);
  }
  const auto packet = PacketContainer(static_cast<uint16_t>(castedEvent.opcode), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
  LOG() << "Injecting packet" << std::endl;
  packetBroker_.injectPacket(packet, direction);
}

// ============================================================================================================================
// ================================================Login process event handling================================================
// ============================================================================================================================

namespace {

std::pair<std::string, std::string> getLoginInfoFromConfig(const config::Config &config) {
  std::unique_lock<std::mutex> configLock(config.mutex());
  const auto &characterName = config.configProto().character_to_login();
  const auto &characterConfig = config.getCharacterConfig(characterName);
  return {characterConfig.username(), characterConfig.password()};
}

} // anonymous namespace

void Bot::handleStateShardIdUpdated() const {
  const auto [username, password] = getLoginInfoFromConfig(config_);
  // We received the server list from the server, try to log in
  const auto loginAuthPacket = packet::building::ClientGatewayLoginRequest::packet(gameData_.divisionInfo().locale, username, password, worldState_.selfState().shardId);
  packetBroker_.injectPacket(loginAuthPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateConnectedToAgentServerUpdated() {
  const auto [username, password] = getLoginInfoFromConfig(config_);
  // Client will try to send auth request, block it
  if (proxy_.blockingOpcode(packet::Opcode::kClientAgentAuthRequest)) {
    throw std::runtime_error("Just connected to AgentServer, but something else blocked client auth packets");
  }
  proxy_.blockOpcode(packet::Opcode::kClientAgentAuthRequest);
  // Send our auth packet
  const auto clientAuthPacket = packet::building::ClientAgentAuthRequest::packet(worldState_.selfState().token, username, password, gameData_.divisionInfo().locale, worldState_.selfState().kMacAddress);
  packetBroker_.injectPacket(clientAuthPacket, PacketContainer::Direction::kClientToServer);
  // Set our state to logging in so that we'll know to block packets from the client if it tries to also login
  worldState_.selfState().loggingIn = true;
}

void Bot::handleLoggedIn() {
  worldState_.selfState().loggingIn = false;
  // Unblock packet
  if (!proxy_.blockingOpcode(packet::Opcode::kClientAgentAuthRequest)) {
    throw std::runtime_error("Just logged in, but we werent blocking the client's auth request");
  }
  proxy_.unblockOpcode(packet::Opcode::kClientAgentAuthRequest);
}

void Bot::handleStateReceivedCaptchaPromptUpdated() const {
  LOG() << "Got captcha. Sending answer\n";
  const auto captchaAnswerPacket = packet::building::ClientGatewayLoginIbuvAnswer::packet(worldState_.selfState().kCaptchaAnswer);
  packetBroker_.injectPacket(captchaAnswerPacket, PacketContainer::Direction::kClientToServer);
}

void Bot::handleStateCharacterListUpdated() const {
  std::string characterName;
  {
    std::unique_lock<std::mutex> configLock(config_.mutex());
    characterName = config_.configProto().character_to_login();
  }
  LOG() << "Char list received: [ ";
  for (const auto &i : worldState_.selfState().characterList) {
    std::cout << i.name << ' ';
  }
  std::cout << "]\n";

  // Search for our character in the character list
  auto it = std::find_if(worldState_.selfState().characterList.begin(), worldState_.selfState().characterList.end(), [&characterName](const packet::structures::CharacterSelection::Character &character) {
    return character.name == characterName;
  });
  if (it == worldState_.selfState().characterList.end()) {
    LOG() << "Unable to find character \"" << characterName << "\"\n";
    return;
  }

  // Found our character, select it
  auto charSelectionPacket = packet::building::ClientAgentCharacterSelectionJoinRequest::packet(characterName);
  packetBroker_.injectPacket(charSelectionPacket, PacketContainer::Direction::kClientToServer);
}

// ============================================================================================================================
// ==================================================Movement event handling===================================================
// ============================================================================================================================

void Bot::handleEntityMovementBegan(const event::EntityMovementBegan &event) {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(event.globalId);
  if (!mobileEntity.moving()) {
    // TODO: Maybe we ought to move this check at the source of the event and get rid of this
    throw std::runtime_error("Got an entity movement began event, but it is not moving");
  }
  onUpdate(&event);
}

void Bot::handleEntityMovementTimerEnded(sro::scalar_types::EntityGlobalId globalId) {
  entity::MobileEntity &mobileEntity = worldState_.getEntity<entity::MobileEntity>(globalId);
  mobileEntity.movementTimerCompleted(eventBroker_);
}

void Bot::handleEntityEnteredGeometry(const event::EntityEnteredGeometry &event) {
  LOG() << "Entity " << event.globalId << " entered geometry" << std::endl;
  onUpdate();
}

void Bot::handleEntityExitedGeometry(const event::EntityExitedGeometry &event) {
  LOG() << "Entity " << event.globalId << " exited geometry" << std::endl;
  onUpdate();
}

// ============================================================================================================================
// ===============================================Character info packet handling===============================================
// ============================================================================================================================

void Bot::handleSpawned(const event::Event *event) {
  LOG() << "Spawned at position " << worldState_.selfState().position() << std::endl;
  onUpdate(event);
}

void Bot::handleVitalsChanged() {
  if (!worldState_.selfState().maxHp() || !worldState_.selfState().maxMp()) {
    // Dont yet know our max
    return;
  }
  onUpdate();
}

void Bot::handleStatesChanged() {
  onUpdate();
}

// ============================================================================================================================
// ===========================================================Skills===========================================================
// ============================================================================================================================

void Bot::handleSkillBegan(const event::SkillBegan &event) {
  if (event.casterGlobalId == worldState_.selfState().globalId) {
    const auto skillName = gameData_.getSkillNameIfExists(event.skillRefId);
    LOG() << "Our skill \"" << (skillName ? *skillName : "UNKNOWN") << "\" began" << std::endl;
    onUpdate(&event);
  }
}

void Bot::handleSkillEnded(const event::SkillEnded &event) {
  if (event.casterGlobalId == worldState_.selfState().globalId) {
    const auto skillName = gameData_.getSkillNameIfExists(event.skillRefId);
    LOG() << "Our skill \"" << (skillName ? *skillName : "UNKNOWN") << "\" ended" << std::endl;
    onUpdate(&event);
  }
}

void Bot::handleSkillCooldownEnded(const event::SkillCooldownEnded &event) {
  const auto skillName = gameData_.getSkillNameIfExists(event.skillRefId);
  LOG() << "Skill " << event.skillRefId << "(" << (skillName ? *skillName : "UNKNOWN") << ") cooldown ended" << std::endl;
  worldState_.selfState().skillEngine.skillCooldownEnded(event.skillRefId);
  onUpdate();
}

// ============================================================================================================================
// ============================================================Misc============================================================
// ============================================================================================================================

void Bot::entitySpawned(const event::EntitySpawned &event) {
  const bool trackingEntity = worldState_.entityTracker().trackingEntity(event.globalId);
  if (!trackingEntity) {
    // TODO: Maybe we ought to move this check at the source of the event and get rid of this
    throw std::runtime_error("Received entity spawned event, but we're not tracking this entity");
  }
  {
    // TODO: Remove. This is a temporary mechanism to measure the maximum visibility range.
    // According to Daxter:
    //  maximum possible should be around 905
    //  given that you can at most see almost 2 blocks across
    const auto &entity = worldState_.getEntity<entity::Entity>(event.globalId);
    const auto distanceToEntity = sro::position_math::calculateDistance2d(worldState_.selfState().position(), entity.position());
    if (distanceToEntity > worldState_.selfState().estimatedVisibilityRange) {
      LOG() << "Bumping up estimated visibility range to " << distanceToEntity << std::endl;
      worldState_.selfState().estimatedVisibilityRange = distanceToEntity;
      std::ofstream estVisRangeFile{kEstVisRangeFilename};
      if (estVisRangeFile) {
        estVisRangeFile << worldState_.selfState().estimatedVisibilityRange;
      }
    }
  }
  onUpdate(&event);
}

void Bot::handleBodyStateChanged(const event::EntityBodyStateChanged &event) {
  if (event.globalId == selfState().globalId) {
    // Our body state changed
    if (selfState().bodyState() == packet::enums::BodyState::kInvisibleGm) {
      // For quicker development, when we spawn in, set ourself as visible and put on a PVP cape
      // Set self as visible
      LOG() << "Setting self as visible" << std::endl;
      const auto setVisiblePacket = packet::building::ClientAgentOperatorRequest::toggleInvisible();
      packetBroker_.injectPacket(setVisiblePacket, PacketContainer::Direction::kClientToServer);

      // LOG() << "Setting free pvp mode" << std::endl;
      // const auto setPvpModePacket = packet::building::ClientAgentFreePvpUpdateRequest::setMode(packet::enums::FreePvpMode::kYellow);
      // packetBroker_.injectPacket(setPvpModePacket, PacketContainer::Direction::kClientToServer);
    }
  }
  onUpdate(&event);
}

void Bot::itemUseTimedOut(const event::ItemUseTimeout &event) {
  LOG() << "Item use timed out!" << std::endl;
  onUpdate(&event);
}

void Bot::handleKnockbackStunEnded() {
  selfState().stunnedFromKnockback = false;
  onUpdate();
}

void Bot::handleKnockdownStunEnded() {
  selfState().stunnedFromKnockdown = false;
  onUpdate();
}

void Bot::handleItemCooldownEnded(const event::ItemCooldownEnded &event) {
  selfState().itemCooldownEnded(event.typeId);
  onUpdate(&event);
}

bool Bot::needToGoToTown() const {
  const auto mpPotionSlots = selfState().inventory.findItemsOfCategory({type_id::categories::kMpPotion});
  if (mpPotionSlots.empty()) {
    LOG() << "Checking if we need to go to town. Have no MP potions" << std::endl;
    return true;
  }
  const auto hpPotionSlots = selfState().inventory.findItemsOfCategory({type_id::categories::kHpPotion});
  if (hpPotionSlots.empty()) {
    LOG() << "Checking if we need to go to town. Have no HP potions" << std::endl;
    return true;
  }
  int hpPotionCount=0;
  for (const auto slot : hpPotionSlots) {
    const auto *item = selfState().inventory.getItem(slot);
    if (item == nullptr) {
      throw std::runtime_error("Expecting hp potion, but item is null");
    }
    const auto *itemAsExp = dynamic_cast<const storage::ItemExpendable*>(item);
    if (itemAsExp == nullptr) {
      throw std::runtime_error("Expecting hp potion, but expendable is null");
    }
    hpPotionCount += itemAsExp->quantity;
  }
  constexpr const int kMinHpCount{10};
  if (hpPotionCount < kMinHpCount) {
    LOG() << "Checking if we need to go to town. Have fewer than " << kMinHpCount << " HP potions" << std::endl;
    return true;
  }
  return false;
}

bool Bot::similarSkillIsAlreadyActive(sro::scalar_types::ReferenceObjectId skillRefId) const {
  const auto &givenSkillData = gameData_.skillData().getSkillById(skillRefId);
  const auto activeBuffs = selfState().activeBuffs();
  for (const auto activeBuffId : activeBuffs) {
    const auto &activeBuffData = gameData_.skillData().getSkillById(activeBuffId);
    if (givenSkillData.actionOverlap == activeBuffData.actionOverlap) {
      // These two cannot be active at the same time
      return true;
    }
  }
  return false;
}

bool Bot::canCastSkill(sro::scalar_types::ReferenceObjectId skillRefId) const {
  const auto &skillData = gameData().skillData().getSkillById(skillRefId);
  // TODO: Keep track if we're wearing a full protector or garment set and reduce the MP requirement by 10%/20% respectively.
  const auto currentMp = selfState().currentMp();
  if (skillData.consumeMP > currentMp ||
      (selfState().maxMp() && skillData.consumeMPRatio > (static_cast<double>(currentMp) / *selfState().maxMp()) * 100)) {
    // Not enough MP to cast.
    LOG() << "Not enough MP to cast skill " << (gameData().getSkillNameIfExists(skillRefId) ? *gameData().getSkillNameIfExists(skillRefId) : std::string("unknown")) << std::endl;
    return false;
  }
  const auto currentHp = selfState().currentHp();
  if (skillData.consumeHP > currentHp ||
      (selfState().maxHp() && skillData.consumeHPRatio > (static_cast<double>(currentHp) / *selfState().maxHp()) * 100)) {
    // Not enough HP to cast.
    LOG() << "Not enough HP to cast skill " << (gameData().getSkillNameIfExists(skillRefId) ? *gameData().getSkillNameIfExists(skillRefId) : std::string("unknown")) << std::endl;
    return false;
  }
  if (selfState().skillEngine.skillIsOnCooldown(skillRefId)) {
    return false;
  }
  if (selfState().stunnedFromKnockback || selfState().stunnedFromKnockdown) {
    // Stunned from KB or KD, cannot use this skill
    // TODO: Maybe there are some skills which can be used while knocked down
    return false;
  }
  return true;
}

std::vector<packet::building::NetworkReadyPosition> Bot::calculatePathToDestination(const sro::Position &destinationPosition) const {
  // Since we can only move to positions on whole integers, find the closest possible point to the destination position while also accounting for the transformation that happens to the packet while being converted to be sent over the network
  const auto networkReadyPos = packet::building::NetworkReadyPosition::roundToNearest(destinationPosition);
  const auto closestDestinationPosition = networkReadyPos.asSroPosition();
  if (sro::position_math::calculateDistance2d(selfState().position(), closestDestinationPosition) <= sqrt(0.5)) {
    // Already at destination
    return {};
  }
  auto pathfindingResultPathToVectorOfPositions = [&](const auto &pathfindingShortestPath) {
    const auto &navmeshTriangulation = gameData().navmeshTriangulation();

    // Get a list of all straight segments
    std::vector<pathfinder::StraightPathSegment*> straightSegments;
    for (const auto &segment : pathfindingShortestPath) {
      pathfinder::StraightPathSegment *straightSegment = dynamic_cast<pathfinder::StraightPathSegment*>(segment.get());
      if (straightSegment != nullptr) {
        straightSegments.push_back(straightSegment);
      }
    }

    // Turn straight segments into a list of waypoints
    std::vector<sro::Position> waypoints;
    // Note: We are ignoring the start of the first segment, since we assume we're already there
    for (int i=0; i<straightSegments.size()-1; ++i) {
      // Find the average between the end of this straight segment and the beginning of the next
      //  Between these two is an arc, which we're ignoring
      // TODO: There is a chance that this yields a bad path
      const auto &point1 = straightSegments[i]->endPoint;
      const auto &point2 = straightSegments[i+1]->startPoint;
      const auto midpoint = pathfinder::math::extendLineSegmentToLength(point1, point2, pathfinder::math::distance(point1, point2)/2.0);
      const auto regionAndPointPair = navmeshTriangulation.transformAbsolutePointIntoRegion({static_cast<float>(midpoint.x()), 0.0f, static_cast<float>(midpoint.y())});
      // TODO: Rounding the position could result in an invalid path
      waypoints.emplace_back(regionAndPointPair.first, std::round(regionAndPointPair.second.x), std::round(regionAndPointPair.second.y), std::round(regionAndPointPair.second.z));
    }

    // Additionally, add the endpoint of the final segment
    const auto finalRegionAndPointPair = navmeshTriangulation.transformAbsolutePointIntoRegion({static_cast<float>(straightSegments.back()->endPoint.x()), 0.0f, static_cast<float>(straightSegments.back()->endPoint.y())});
    waypoints.emplace_back(finalRegionAndPointPair.first, std::round(finalRegionAndPointPair.second.x), std::round(finalRegionAndPointPair.second.y), std::round(finalRegionAndPointPair.second.z));

    // Remove duplicates
    auto newEndIt = std::unique(waypoints.begin(), waypoints.end());
    if (newEndIt != waypoints.end()) {
      waypoints.erase(newEndIt, waypoints.end());
    }
    return waypoints;
  };

  auto breakUpLongMovements = [](std::vector<sro::Position> &waypoints) {
    auto tooFar = [](const auto &srcWaypoint, const auto &destWaypoint) {
      // The difference between a pair of xOffsets must be <= 1920.
      // The difference between a pair of zOffsets must be <= 1920.
      const auto [dx, dz] = sro::position_math::calculateOffset2d(srcWaypoint, destWaypoint);
      return (std::abs(dx) > sro::game_constants::kRegionWidth ||
              std::abs(dz) > sro::game_constants::kRegionHeight);
    };
    auto splitWaypoints = [](const auto &srcWaypoint, const auto &destWaypoint) {
      const auto [dx, dz] = sro::position_math::calculateOffset2d(srcWaypoint, destWaypoint);
      if (std::abs(dx) > std::abs(dz)) {
        const double ratio = static_cast<double>(sro::game_constants::kRegionWidth) / std::abs(dx);
        const auto newDxOffset = (dx > 0 ? 1 : -1) * sro::game_constants::kRegionWidth;
        const double newDzOffset = dz * ratio;
        return sro::position_math::createNewPositionWith2dOffset(destWaypoint, -newDxOffset, -newDzOffset);
      } else {
        const double ratio = static_cast<double>(sro::game_constants::kRegionWidth) / std::abs(dz);
        const auto newDxOffset = dx * ratio;
        const double newDzOffset = (dz > 0 ? 1 : -1) * sro::game_constants::kRegionWidth;
        return sro::position_math::createNewPositionWith2dOffset(destWaypoint, -newDxOffset, -newDzOffset);
      }
    };
    for (int i=waypoints.size()-1; i>0;) {
      if (tooFar(waypoints.at(i-1), waypoints.at(i))) {
        // Pick a point that is the maxmimum distance possible away from waypoints[i] and insert it before waypoints[i]
        const auto newWaypoint = splitWaypoints(waypoints.at(i-1), waypoints.at(i));
        waypoints.insert(waypoints.begin()+i, newWaypoint);
      } else {
        --i;
      }
    }
  };

  auto convertWaypointsToNetworkReadyPoints = [](const std::vector<sro::Position> &waypoints) {
    std::vector<packet::building::NetworkReadyPosition> result;
    result.reserve(waypoints.size());
    for (const auto &pos : waypoints) {
      result.emplace_back(packet::building::NetworkReadyPosition::roundToNearest(pos));
    }
    return result;
  };

  constexpr const double kAgentRadius{7.23};
  pathfinder::Pathfinder<navmesh::triangulation::NavmeshTriangulation> pathfinder(gameData().navmeshTriangulation(), kAgentRadius);
  try {
    const auto currentPosition = selfState().position();
    const sro::math::Vector3 currentPositionPoint(currentPosition.xOffset(), currentPosition.yOffset(), currentPosition.zOffset());
    const auto navmeshCurrentPosition = gameData().navmeshTriangulation().transformRegionPointIntoAbsolute(currentPositionPoint, currentPosition.regionId());

    const sro::math::Vector3 destinationPositionPoint(closestDestinationPosition.xOffset(), closestDestinationPosition.yOffset(), closestDestinationPosition.zOffset());
    const auto navmeshDestinationPosition = gameData().navmeshTriangulation().transformRegionPointIntoAbsolute(destinationPositionPoint, destinationPosition.regionId());

    // TODO: If the src or dest positions are overlapping with a constraint, we need to add an extra point.
    const auto pathfindingResult = pathfinder.findShortestPath(navmeshCurrentPosition, navmeshDestinationPosition);
    const auto &path = pathfindingResult.shortestPath;
    if (path.empty()) {
      throw std::runtime_error("Found empty path");
    }
    auto waypoints = pathfindingResultPathToVectorOfPositions(path);
    // Add our own position to the beginning of this list so that we can break up the distance if it's too far.
    waypoints.insert(waypoints.begin(), currentPosition);
    breakUpLongMovements(waypoints);
    return convertWaypointsToNetworkReadyPoints(waypoints);
  } catch (std::exception &ex) {
    throw std::runtime_error("Cannot find path with pathfinder: \""+std::string(ex.what())+"\"");
  }
}