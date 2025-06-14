#include "checkpointManager.hpp"
#include "rl/ai/deepLearningIntelligence.hpp"
#include "ui/rlUserInterface.hpp"

#include <silkroad_lib/file_util.hpp>

// Tracy
#include <common/TracySystem.hpp>

#include <absl/log/log.h>

#include <fstream>
#include <filesystem>

using namespace proto;

namespace rl {

CheckpointManager::CheckpointManager(ui::RlUserInterface &rlUserInterface) : rlUserInterface_(rlUserInterface) {
  std::unique_lock lock(registryMutex_);
  LOG(INFO) << "Constructing CheckpointManager";
  const std::filesystem::path appDataPath = sro::file_util::getAppDataPath();
  const std::filesystem::path checkpointsDir = appDataPath / kCheckpointDirectoryName;
  if (!std::filesystem::exists(checkpointsDir)) {
    std::filesystem::create_directory(checkpointsDir);
  }

  LOG(INFO) << "Loading checkpoint registry from " << appDataPath;
  std::ifstream checkpointRegistryFile(appDataPath / kCheckpointRegistryFilename);
  if (checkpointRegistryFile.is_open()) {
    LOG(INFO) << "Successfully opened checkpoint registry file";
    std::string checkpointRegistryFileContents((std::istreambuf_iterator<char>(checkpointRegistryFile)), std::istreambuf_iterator<char>());
    if (!checkpointRegistry_.ParseFromString(checkpointRegistryFileContents)) {
      LOG(ERROR) << "Failed to parse checkpoint registry from file";
    }
  } else {
    // File does not exist. Create an empty one.
    LOG(INFO) << "Checkpoint registry file does not exist. Creating empty one.";
    saveCurrentRegistryNoLock();
  }

  bool registryModified = false;
  auto *checkpoints = checkpointRegistry_.mutable_checkpoints();
  for (auto it = checkpoints->begin(); it != checkpoints->end();) {
    const std::filesystem::path modelPath = it->model_checkpoint_path();
    const std::filesystem::path targetPath = it->target_model_checkpoint_path();
    const std::filesystem::path optimizerPath = it->optimizer_checkpoint_path();
    if (!std::filesystem::exists(modelPath) || !std::filesystem::exists(targetPath) ||
        !std::filesystem::exists(optimizerPath)) {
      LOG(WARNING) << "Removing missing checkpoint \"" << it->checkpoint_name() << "\" from registry";
      it = checkpoints->erase(it);
      registryModified = true;
      continue;
    }

    const std::filesystem::path newBasePath = checkpointsDir / it->checkpoint_name();
    const std::filesystem::path newModelPath = newBasePath.string() + "_model";
    const std::filesystem::path newTargetPath = newBasePath.string() + "_target_model";
    const std::filesystem::path newOptimizerPath = newBasePath.string() + "_optimizer_state";
    if (modelPath != newModelPath) {
      try {
        std::filesystem::rename(modelPath, newModelPath);
        std::filesystem::rename(targetPath, newTargetPath);
        std::filesystem::rename(optimizerPath, newOptimizerPath);
        it->set_model_checkpoint_path(newModelPath.string());
        it->set_target_model_checkpoint_path(newTargetPath.string());
        it->set_optimizer_checkpoint_path(newOptimizerPath.string());
        registryModified = true;
      } catch (const std::filesystem::filesystem_error &e) {
        LOG(ERROR) << "Failed to move checkpoint \"" << it->checkpoint_name() << "\": " << e.what();
      }
    }
    ++it;
  }

  if (registryModified) {
    saveCurrentRegistryNoLock();
    rlUserInterface_.sendCheckpointList(getCheckpointNames());
  }
}

CheckpointManager::~CheckpointManager() {
  if (checkpointingThread_.joinable()) {
    checkpointingThread_.join();
  }
}

void CheckpointManager::saveCheckpoint(const std::string &checkpointName, rl::JaxInterface &jaxInterface, int stepCount, bool overwrite) {
  const bool checkpointAlreadyExists = checkpointExists(checkpointName);
  if (!overwrite && checkpointAlreadyExists) {
    throw std::runtime_error("Trying to create a checkpoint which already exists");
  }
  const std::filesystem::path checkpointsDir = sro::file_util::getAppDataPath() / kCheckpointDirectoryName;
  if (!std::filesystem::exists(checkpointsDir)) {
    std::filesystem::create_directory(checkpointsDir);
  }

  const std::string fullCheckpointPath = (checkpointsDir / checkpointName).string();
  if (checkpointAlreadyExists) {
    LOG(INFO) << "Overwriting checkpoint \"" << checkpointName << "\" at path \"" << fullCheckpointPath << "\"";
  } else {
    LOG(INFO) << "Saving checkpoint \"" << checkpointName << "\" to path \"" << fullCheckpointPath << "\"";
  }
  const std::string modelCheckpointPath = fullCheckpointPath + "_model";
  const std::string targetModelCheckpointPath = fullCheckpointPath + "_target_model";
  const std::string optimizerCheckpointPath = fullCheckpointPath + "_optimizer_state";
  if (!checkpointAlreadyExists) {
    std::unique_lock lock(registryMutex_);
    rl_checkpointing::Checkpoint *newCheckpointProto = checkpointRegistry_.add_checkpoints();
    newCheckpointProto->set_checkpoint_name(checkpointName);
    newCheckpointProto->set_model_checkpoint_path(modelCheckpointPath);
    newCheckpointProto->set_target_model_checkpoint_path(targetModelCheckpointPath);
    newCheckpointProto->set_optimizer_checkpoint_path(optimizerCheckpointPath);
    newCheckpointProto->set_step_count(stepCount);
    saveCurrentRegistryNoLock();
  } else {
    // Need to update the step count.
    std::unique_lock lock(registryMutex_);
    for (rl_checkpointing::Checkpoint &checkpoint : *checkpointRegistry_.mutable_checkpoints()) {
      if (checkpoint.checkpoint_name() == checkpointName) {
        checkpoint.set_step_count(stepCount);
        break;
      }
    }
    saveCurrentRegistryNoLock();
  }
  if (checkpointingThread_.joinable()) {
    VLOG(1) << "Checkpointing thread is already running. Waiting for it to finish.";
    checkpointingThread_.join();
  }
  checkpointingThread_ = std::thread([this, checkpointName, modelCheckpointPath, targetModelCheckpointPath, optimizerCheckpointPath, &jaxInterface]() {
    // TODO: This class should entirely be run on another thread. Any request should get queued to that thread.
    tracy::SetThreadName("CheckpointManager");
    rlUserInterface_.sendSavingCheckpoint();
    jaxInterface.saveCheckpoint(modelCheckpointPath, targetModelCheckpointPath, optimizerCheckpointPath);
    rlUserInterface_.sendCheckpointList(getCheckpointNames());
    LOG(INFO) << "Saved checkpoint \"" << checkpointName << "\"";
  });
}

bool CheckpointManager::checkpointExists(const std::string &checkpointName) const {
  std::unique_lock lock(registryMutex_);
  for (const rl_checkpointing::Checkpoint &checkpoint : checkpointRegistry_.checkpoints()) {
    if (checkpoint.checkpoint_name() == checkpointName) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> CheckpointManager::getCheckpointNames() const {
  std::unique_lock lock(registryMutex_);
  std::vector<std::string> checkpointNames;
  for (const rl_checkpointing::Checkpoint &checkpoint : checkpointRegistry_.checkpoints()) {
    checkpointNames.push_back(checkpoint.checkpoint_name());
  }
  return checkpointNames;
}

CheckpointValues CheckpointManager::loadCheckpoint(const std::string &checkpointName, rl::JaxInterface &jaxInterface) {
  std::unique_lock lock(registryMutex_);
  for (const rl_checkpointing::Checkpoint &checkpoint : checkpointRegistry_.checkpoints()) {
    if (checkpoint.checkpoint_name() == checkpointName) {
      CheckpointValues result;
      const std::string modelCheckpointPath = checkpoint.model_checkpoint_path();
      const std::string targetModelCheckpointPath = checkpoint.target_model_checkpoint_path();
      const std::string optimizerCheckpointPath = checkpoint.optimizer_checkpoint_path();
      LOG(INFO) << "Loading checkpoint \"" << checkpointName << "\" from paths \"" << modelCheckpointPath << "\", \"" << targetModelCheckpointPath << "\", and \"" << optimizerCheckpointPath << "\"";
      jaxInterface.loadCheckpoint(modelCheckpointPath, targetModelCheckpointPath, optimizerCheckpointPath);
      result.stepCount = checkpoint.step_count();
      // Notify the UI that a checkpoint has been loaded
      rlUserInterface_.sendCheckpointLoaded(checkpointName);
      return result;
    }
  }
  throw std::runtime_error("Checkpoint not found");
}

void CheckpointManager::deleteCheckpoints(const std::vector<std::string> &checkpointNames) {
  {
    std::unique_lock lock(registryMutex_);
    for (const std::string &checkpointName : checkpointNames) {
      auto it = std::find_if(checkpointRegistry_.mutable_checkpoints()->begin(), checkpointRegistry_.mutable_checkpoints()->end(),
                            [&checkpointName](const rl_checkpointing::Checkpoint &checkpoint) { return checkpoint.checkpoint_name() == checkpointName; });
      if (it == checkpointRegistry_.mutable_checkpoints()->end()) {
        LOG(WARNING) << "Checkpoint \"" << checkpointName << "\" not found";
        continue;
      }
      // Delete the actual checkpoint files from disk.
      VLOG(1) << "Deleting checkpoint \"" << checkpointName << "\" from registry, which is comprised of:";
      VLOG(1) << "  - Model checkpoint path: " << it->model_checkpoint_path();
      VLOG(1) << "  - Target model checkpoint path: " << it->target_model_checkpoint_path();
      VLOG(1) << "  - Optimizer checkpoint path: " << it->optimizer_checkpoint_path();
      std::uintmax_t filesRemoved;
      filesRemoved = std::filesystem::remove_all(it->model_checkpoint_path());
      if (filesRemoved == 0) {
        LOG(WARNING) << "Failed to delete model checkpoint files";
      }
      filesRemoved = std::filesystem::remove_all(it->target_model_checkpoint_path());
      if (filesRemoved == 0) {
        LOG(WARNING) << "Failed to delete target model checkpoint files";
      }
      filesRemoved = std::filesystem::remove_all(it->optimizer_checkpoint_path());
      if (filesRemoved == 0) {
        LOG(WARNING) << "Failed to delete optimizer checkpoint files";
      }
      // Remove the checkpoint from the registry.
      checkpointRegistry_.mutable_checkpoints()->erase(it);
      LOG(INFO) << "Deleted checkpoint \"" << checkpointName << "\"";
    }
    LOG(INFO) << "Saving checkpoint registry after deleting checkpoints";
    saveCurrentRegistryNoLock();
  }
  LOG(INFO) << "Sending new checkpoint list to UI";
  rlUserInterface_.sendCheckpointList(getCheckpointNames());
  LOG(INFO) << "Sent new checkpoint list to UI";
}

void CheckpointManager::saveCurrentRegistryNoLock() {
  std::ofstream checkpointRegistryFile(sro::file_util::getAppDataPath() / kCheckpointRegistryFilename);
  if (!checkpointRegistryFile.is_open()) {
    throw std::runtime_error("Failed to open/create checkpoint registry file");
  }
  if (!checkpointRegistry_.SerializeToOstream(&checkpointRegistryFile)) {
    throw std::runtime_error("Failed to serialize checkpoint registry to file");
  }
}

} // namespace rl