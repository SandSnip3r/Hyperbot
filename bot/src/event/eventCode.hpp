#ifndef EVENT_EVENT_CODE_HPP_
#define EVENT_EVENT_CODE_HPP_

#include <string_view>

#define EVENT_EVENTCODE_LIST(F) \
  F(SelfSpawned) \
  F(CosSpawned) \
  F(InternalItemCooldownEnded) \
  F(ItemCooldownEnded) \
  F(EntityHpChanged) \
  F(EntityMpChanged) \
  F(MaxHpMpChanged) \
  F(StatsChanged) \
  F(EntityStatesChanged) \
  F(InternalSkillCooldownEnded) \
  F(SkillCooldownEnded) \
  F(ItemMoved) \
  F(ItemMoveFailed) \
  F(SkillCastAboutToEnd) \
  F(ItemUseSuccess) \
  F(ItemUseFailed) \
  F(InjectPacket) \
  F(RequestStartTraining) \
  F(RequestStopTraining) \
  F(TrainingStarted) \
  F(TrainingStopped) \
  F(EntityDeselected) \
  F(EntitySelected) \
  F(NpcTalkStart) \
  F(StorageInitialized) \
  F(GuildStorageInitialized) \
  F(RepairSuccessful) \
  F(InventoryGoldUpdated) \
  F(StorageGoldUpdated) \
  F(GuildStorageGoldUpdated) \
  F(CharacterLevelUpdated) \
  F(CharacterSkillPointsUpdated) \
  F(CharacterAvailableStatPointsUpdated) \
  F(CharacterExperienceUpdated) \
  F(EnteredNewRegion) \
  F(EntitySpawned) \
  F(EntityDespawned) \
  F(EntityMovementEnded) \
  F(EntityOwnershipRemoved) \
  F(StateMachineCreated) \
  F(StateMachineDestroyed) \
  F(SkillBegan) \
  F(SkillEnded) \
  F(DealtDamage) \
  F(SkillFailed) \
  F(BuffAdded) \
  F(BuffRemoved) \
  F(CommandError) \
  F(CommandSkipped) \
  F(SkillCastTimeout) \
  F(EntityMovementBegan) \
  F(StateMachineActiveTooLong) \
  F(EntityMovementTimerEnded) \
  F(EntityPositionUpdated) \
  F(EntityNotMovingAngleChanged) \
  F(EntityBodyStateChanged) \
  F(EntityLifeStateChanged) \
  F(EntityEnteredGeometry) \
  F(EntityExitedGeometry) \
  F(TrainingAreaSet) \
  F(TrainingAreaReset) \
  F(KnockedBack) \
  F(KnockedDown) \
  F(KnockbackStunEnded) \
  F(KnockdownStunEnded) \
  F(MovementRequestTimedOut) \
  F(WalkingPathUpdated) \
  F(InventoryItemUpdated) \
  F(HwanPointsUpdated) \
  F(AlchemyCompleted) \
  F(AlchemyTimedOut) \
  F(GmCommandTimedOut) \
  F(ChatReceived) \
  F(SetCurrentPositionAsTrainingCenter) \
  F(ResurrectOption) \
  F(LearnMasterySuccess) \
  F(LearnSkillSuccess) \
  F(LearnSkillError) \
  F(Timeout) \
  F(LoginCompleted) \
  F(RlStartPvp) \
  F(StateUpdated) \
  F(ServerAuthSuccess) \
  F(GatewayPatchResponseReceived) \
  F(ShardListReceived) \
  F(GatewayLoginResponseReceived) \
  F(ConnectedToAgentServer) \
  F(CharacterListReceived) \
  F(IbuvChallengeReceived) \
  F(CharacterSelectionJoinSuccess) \
  F(OperatorRequestSuccess) \
  F(OperatorRequestError) \
  F(EquipCountdownStart) \
  F(FreePvpUpdateSuccess) \
  F(PvpManagerReadyForAssignment) \
  F(BeginPvp) \
  F(ReadyForPvp) \
  F(DispelSuccess) \
  F(ClientDied) \
  F(GameReset) \
  F(RlUiStartTraining) \
  F(RlUiStopTraining) \
  F(RlUiRequestCheckpointList) \
  F(RlUiSaveCheckpoint) \
  F(RlUiLoadCheckpoint) \
  F(RlUiDeleteCheckpoints) \
  F(Dummy)

namespace event {

enum class EventCode {
#define F(name) k##name,
  EVENT_EVENTCODE_LIST(F)
#undef F
};

std::string_view toString(EventCode eventCode);

} // namespace event

#endif // EVENT_EVENT_CODE_HPP_