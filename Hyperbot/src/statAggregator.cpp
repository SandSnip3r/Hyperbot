#include "statAggregator.hpp"

#include "../../common/Common.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <absl/log/log.h>

#include <chrono>
#include <filesystem>

proto::stats::StatEvent createStatEvent();
void serializeKilledEntity(const entity::Entity *entity, proto::entity::Entity &protoEntity);

StatAggregator::StatAggregator(const state::WorldState &worldState, broker::EventBroker &eventBroker) : worldState_(worldState), eventBroker_(eventBroker) {/* 
  auto eventHandleFunction = std::bind(&StatAggregator::handleEvent, this, std::placeholders::_1);
  eventBroker_.subscribeToEvent(event::EventCode::kSelfSpawned, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kTrainingStarted, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kTrainingStopped, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kEntityLifeStateChanged, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kKilledEntity, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kDealtDamage, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterSkillPointsUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterExperienceUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryGoldUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kCharacterLevelUpdated, eventHandleFunction);
  eventBroker_.subscribeToEvent(event::EventCode::kInventoryUpdated, eventHandleFunction);
 */}

void StatAggregator::handleEvent(const event::Event *event) {/* 
  if (event != nullptr && event->eventCode == event::EventCode::kSelfSpawned) {
    if (worldState_.selfState().name != characterName_) {
      // We must have just logged in
      if (characterName_ != "") {
        throw std::runtime_error("We spawned, but now with a different name");
      }
      try {
        initialize(worldState_.selfState().name);
      } catch (std::exception &ex) {
        LOG(INFO) << "Error while initializing " << ex.what();
      }
    }
  }

  if (!initialized_) {
    return;
  }

  if (event != nullptr) {
    // Convert event to proto.
    const auto eventCode = event->eventCode;
    if (eventCode == event::EventCode::kTrainingStarted) {
      auto protoEvent = createStatEvent();
      protoEvent.mutable_start_training();
      writeEventToStatFile(protoEvent);
    } else if (eventCode == event::EventCode::kTrainingStopped) {
      auto protoEvent = createStatEvent();
      protoEvent.mutable_stop_training();
      writeEventToStatFile(protoEvent);
    } else if (const auto *entityLifeStateChanged = dynamic_cast<const event::EntityLifeStateChanged*>(event)) {
      if (entityLifeStateChanged->globalId == worldState_.selfState().globalId && worldState_.selfState().lifeState == sro::entity::LifeState::kDead) {
        // We died.
        // TODO: Who killed us?
        auto protoEvent = createStatEvent();
        protoEvent.mutable_died();
        writeEventToStatFile(protoEvent);
      }
    } else if (const auto *killedEntity = dynamic_cast<const event::KilledEntity*>(event)) {
      auto protoEvent = createStatEvent();
      const auto *entity = worldState_.getEntity(killedEntity->targetId);
      auto *entityKilledEvent = protoEvent.mutable_entity_killed();
      serializeKilledEntity(entity, *entityKilledEvent);
      writeEventToStatFile(protoEvent);
    } else if (const auto *dealtDamage = dynamic_cast<const event::DealtDamage*>(event)) {
      if (dealtDamage->sourceId == worldState_.selfState().globalId) {
        auto protoEvent = createStatEvent();
        protoEvent.set_damage_dealt(dealtDamage->damageAmount);
        writeEventToStatFile(protoEvent);
      }
    } else if (eventCode == event::EventCode::kCharacterSkillPointsUpdated) {
      auto protoEvent = createStatEvent();
      protoEvent.set_skillpoint_updated(worldState_.selfState().getSkillPoints()*400 + worldState_.selfState().getCurrentSpExperience());
      writeEventToStatFile(protoEvent);
    } else if (eventCode == event::EventCode::kCharacterExperienceUpdated) {
      // One event for skill point experience.
      auto protoEventSp = createStatEvent();
      protoEventSp.set_skillpoint_updated(worldState_.selfState().getSkillPoints()*400 + worldState_.selfState().getCurrentSpExperience());
      writeEventToStatFile(protoEventSp);
      // A second event for regular experience.
      auto protoEventExp = createStatEvent();
      protoEventExp.set_experience_updated(worldState_.selfState().getCurrentExperience());
      writeEventToStatFile(protoEventExp);
    } else if (eventCode == event::EventCode::kInventoryGoldUpdated) {
      auto protoEvent = createStatEvent();
      protoEvent.set_gold_updated(worldState_.selfState().getGold());
      writeEventToStatFile(protoEvent);
    } else if (eventCode == event::EventCode::kCharacterLevelUpdated) {
      auto protoEvent = createStatEvent();
      protoEvent.set_level_updated(worldState_.selfState().getCurrentLevel());
      writeEventToStatFile(protoEvent);
    } else if (eventCode == event::EventCode::kInventoryUpdated) {
      const auto &castedEvent = dynamic_cast<const event::InventoryUpdated&>(*event);
      if (!castedEvent.srcSlotNum && castedEvent.destSlotNum) {
        // Something came into our inventory.
        auto protoEvent = createStatEvent();
        protoEvent.set_drop_gained(worldState_.selfState().inventory.getItem(*castedEvent.destSlotNum)->refItemId);
        writeEventToStatFile(protoEvent);
      }
    }
  }

 */}

void StatAggregator::printParsedFiles(const proto::stats::StatFileRegistry &registry) const {/* 
  for (const auto &i : registry.character_entries()) {
    const auto &charName =  i.first;
    LOG(INFO) << "\"" << charName << "\":";;
    for (const auto &fileData : i.second.files()) {
      if (fileData.proto_version_number() != kVersionNum) {
        LOG(INFO) << "    File \"" << fileData.filename() << "\" has wrong version number: " << fileData.proto_version_number();
        continue;
      }
      const auto statsPath = getAppDataPath() / fileData.filename();
      std::ifstream statsFile(statsPath, std::ios::binary);
      if (!statsFile) {
        LOG(INFO) << "    Cannot open \"" << statsPath.string() << "\"";
        continue;
      }
      google::protobuf::io::IstreamInputStream isistream_(&statsFile);
      google::protobuf::io::CodedInputStream input(&isistream_);
      while (1) {
        size_t size;
        bool readResult = input.ReadRaw(reinterpret_cast<char*>(&size), sizeof(size));
        if (!readResult) {
          // Could not read size; must be done.
          break;
        }
        const auto limit = input.PushLimit(size);
        proto::stats::StatEvent event;
        bool good{true};
        bool res = event.ParseFromCodedStream(&input);
        if (!res) {
          LOG(INFO) << "     Did not read message";
          break;
        }
        if (!input.ConsumedEntireMessage()) {
          LOG(INFO) << "     Did not consume entire message!!";
          break;
        }
        LOG(INFO) << "    " << event.DebugString();
        input.PopLimit(limit);
      }
    }
  }
 */}

void StatAggregator::initialize(const std::string &characterName) {/* 
  characterName_ = characterName;
  filename_ = generateFilename();

  // Open the registry file
  // TODO: Find the registry filename elsewhere
  const auto appDataDirectoryPath = getAppDataPath();
  const auto statRegistryPath = appDataDirectoryPath / "stat_registry";

  proto::stats::StatFileRegistry registry;
  {
    std::ifstream registryInFile(statRegistryPath, std::ios::binary);
    if (registryInFile) {
      // Parse in the file contents into the proto
      registry.ParseFromIstream(&registryInFile);
    }
    // printParsedFiles(registry);
    // Else, file doesn't exist, use a default StatFileRegistry
  }

  // Add our filename to the list
  // TODO: Make sure that the entry exists for this character if [] doesn't default construct one like std::map would.
  auto &characterEntry = (*registry.mutable_character_entries())[characterName_];
  auto *newFileData = characterEntry.add_files();
  newFileData->set_proto_version_number(kVersionNum);
  newFileData->set_filename(filename_);

  // Write the updated data back to the file
  // TODO: Does our index from reading influence the index at which we write?
  {
    std::ofstream registryOutFile(statRegistryPath, std::ios::binary);
    if (!registryOutFile) {
      throw std::runtime_error("Cannot open registry file for writing");
    }
    registry.SerializeToOstream(&registryOutFile);
  }

  // Open our actual file for streaming
  const auto statsPath = appDataDirectoryPath / filename_;
  statFile_ = std::ofstream(statsPath, std::ios::binary);
  if (!statFile_) {
    throw std::runtime_error("Unable to open stats file " + statsPath.string());
  }

  initialized_ = true;
 */}

std::string StatAggregator::generateFilename() const {/* 
  // Use a hash of the character name combined with the current time.
  const auto currentTime = std::chrono::system_clock::now();
  const auto hashOfCharacterName = std::hash<std::string>{}(characterName_);
  return std::to_string(std::chrono::duration_cast<std::chrono::seconds>(currentTime.time_since_epoch()).count()) + "_" + std::to_string(hashOfCharacterName);
 */
  return {};
}

void StatAggregator::writeEventToStatFile(const proto::stats::StatEvent &event) {/* 
  const size_t eventDataSize = event.ByteSizeLong();
  // First write the size of the message.
  statFile_.write(reinterpret_cast<const char*>(&eventDataSize), sizeof(eventDataSize));
  // Now write the message.
  event.SerializeToOstream(&statFile_);
  statFile_.flush();
  // LOG(INFO) << "Writing message \"" << event.DebugString() << "\" which has bin size " << eventDataSize;
 */}

proto::stats::StatEvent createStatEvent() {
  // Get the current time as a time_point
  std::chrono::system_clock::time_point currentTime = std::chrono::system_clock::now();

  // Convert the time_point to seconds and nanoseconds
  std::chrono::seconds seconds = std::chrono::time_point_cast<std::chrono::seconds>(currentTime).time_since_epoch();
  std::chrono::nanoseconds nanos = std::chrono::time_point_cast<std::chrono::nanoseconds>(currentTime).time_since_epoch();

  // Create an instance of the Timestamp message
  proto::stats::StatEvent event;
  google::protobuf::Timestamp *timestamp = event.mutable_timestamp();

  // Set the seconds and nanos fields
  timestamp->set_seconds(seconds.count());
  timestamp->set_nanos(nanos.count() % 1000000000);

  return event;
}

void serializeKilledEntity(const entity::Entity *entity, proto::entity::Entity &protoEntity) {
  protoEntity.set_reference_id(entity->globalId);
  switch (entity->entityType()) {
    case entity::EntityType::kCharacter:
      protoEntity.mutable_character();
      break;
    case entity::EntityType::kPlayerCharacter:
      protoEntity.mutable_player_character();
      break;
    case entity::EntityType::kNonplayerCharacter:
      protoEntity.mutable_nonplayer_character();
      break;
    case entity::EntityType::kMonster: {
      auto *monsterKilledEvent = protoEntity.mutable_monster();
      const auto *entityAsMonster = dynamic_cast<const entity::Monster*>(entity);
      if (entityAsMonster == nullptr) {
        throw std::runtime_error("Entity type is monster, but cannot cast entity to monster type");
      }
      switch (entityAsMonster->rarity) {
        case sro::entity::MonsterRarity::kGeneral:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kGeneral);
          break;
        case sro::entity::MonsterRarity::kChampion:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kChampion);
          break;
        case sro::entity::MonsterRarity::kUnique:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kUnique);
          break;
        case sro::entity::MonsterRarity::kGiant:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kGiant);
          break;
        case sro::entity::MonsterRarity::kTitan:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kTitan);
          break;
        case sro::entity::MonsterRarity::kElite:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kElite);
          break;
        case sro::entity::MonsterRarity::kEliteStrong:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kEliteStrong);
          break;
        case sro::entity::MonsterRarity::kUnique2:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kUnique2);
          break;
        case sro::entity::MonsterRarity::kGeneralParty:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kGeneralParty);
          break;
        case sro::entity::MonsterRarity::kChampionParty:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kChampionParty);
          break;
        case sro::entity::MonsterRarity::kUniqueParty:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kUniqueParty);
          break;
        case sro::entity::MonsterRarity::kGiantParty:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kGiantParty);
          break;
        case sro::entity::MonsterRarity::kTitanParty:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kTitanParty);
          break;
        case sro::entity::MonsterRarity::kEliteParty:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kEliteParty);
          break;
        case sro::entity::MonsterRarity::kEliteStrongParty:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kEliteStrongParty);
          break;
        case sro::entity::MonsterRarity::kUnique2Party:
          monsterKilledEvent->set_rarity(proto::entity::MonsterRarity::kUnique2Party);
          break;
        default:
          throw std::runtime_error("Unknown monster rarity");
      }
      break;
    }
    default:
      throw std::runtime_error("Killed unknown entity type");
  }
}