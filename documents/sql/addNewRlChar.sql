-- Add masteries for a specific character
DECLARE @mastery_id INT
DECLARE @mastery_level INT = 32
DECLARE @charid INT = 6747

DECLARE @mastery_ids TABLE (val INT)
INSERT INTO @mastery_ids (val) VALUES (257), (274), (275)

DECLARE cur CURSOR FOR SELECT val FROM @mastery_ids
OPEN cur

FETCH NEXT FROM cur INTO @mastery_id
WHILE @@FETCH_STATUS = 0
BEGIN
    EXEC [dbo].[_skill_manage] @job = 1, @charid = @charid, @param1 = @mastery_id, @param2 = @mastery_level;
    FETCH NEXT FROM cur INTO @mastery_id
END

CLOSE cur
DEALLOCATE cur

-- Add skills for a specific character
DECLARE @skill_id INT
DECLARE @charid INT = 6747

DECLARE @skill_ids TABLE (val INT)
INSERT INTO @skill_ids (val) VALUES (28), (37), (114), (131), (298), (300), (322), (339), (371), (554), (588), (610), (644), (691), (1253), (1256), (1271), (1272), (1281), (1315), (1335), (1343), (1360), (1377), (1380), (1398), (1399), (1410), (1421), (1441), (1449), (1501), (8312), (21209), (30577)

DECLARE cur CURSOR FOR SELECT val FROM @skill_ids
OPEN cur

FETCH NEXT FROM cur INTO @skill_id
WHILE @@FETCH_STATUS = 0
BEGIN
    EXEC [dbo].[_skill_manage] @job = 0, @charid = @charid, @param1 = 0, @param2 = @skill_id;
    FETCH NEXT FROM cur INTO @skill_id
END

CLOSE cur
DEALLOCATE cur

----------------------------------------------------------------------------------------

USE [SRO_VT_SHARD]
GO

DECLARE	@return_value int

EXEC	@return_value = [dbo].[_AddNewRLChar]
		@UserJID = 1068,
		@RefCharID = 1911,
		@CharName = N'RL_40',
		@CharScale = 68,
		@StartRegionID = 25256,
		@StartPos_X = 977,
		@StartPos_Y = 0,
		@StartPos_Z = 11,
		@DefaultTeleport = 19554,
		@InventorySize = 109,
		@RefMailID = 656,
		@RefPantsID = 692,
		@RefBootsID = 764,
		@RefHelmID = 584,
		@RefShoulderGuardID = 620,
		@RefGauntletID = 731,
		@RefWeaponID = 119,
		@RefShield = 263,
		@DurMail = 50,
		@DurPants = 50,
		@DurBoots = 50,
		@DurHelm = 50,
		@DurShoulderGuard = 50,
		@DurGauntlet = 50,
		@DurWeapon = 50,
		@DurShield = 50,
		@DefaultArrow = 0,
		@RefEarringID = 1844,
		@RefNecklaceID = 1880,
		@RefLeftRingID = 1811,
		@RefRightRingID = 1811,
		@Strength = 144,
		@Intellect = 51,
		@CurLevel = 32,
		@MaxLevel = 32,
		@ExpOffset = 200000,
		@SExpOffset = 200,
		@RemainGold = 1000000000,
		@RemainSkillPoint = 1000000

DECLARE @SourceCharID INT = 6745;
DECLARE @DestinationCharID INT = @return_value;

-- Step 1: Delete existing config rows for the destination character
DELETE FROM [SRO_VT_SHARD].[dbo].[_ClientConfig]
WHERE CharID = @DestinationCharID;

-- Step 2: Insert copied config rows from source character
INSERT INTO [SRO_VT_SHARD].[dbo].[_ClientConfig] (
    [CharID],
    [ConfigType],
    [SlotSeq],
    [SlotType],
    [Data]
)
SELECT 
    @DestinationCharID AS CharID,
    [ConfigType],
    [SlotSeq],
    [SlotType],
    [Data]
FROM [SRO_VT_SHARD].[dbo].[_ClientConfig]
WHERE CharID = @SourceCharID;

-- Add masteries for a specific character
DECLARE @mastery_id INT
DECLARE @mastery_level INT = 32

DECLARE @mastery_ids TABLE (val INT)
INSERT INTO @mastery_ids (val) VALUES (257), (274), (275)

DECLARE cur CURSOR FOR SELECT val FROM @mastery_ids
OPEN cur

FETCH NEXT FROM cur INTO @mastery_id
WHILE @@FETCH_STATUS = 0
BEGIN
    EXEC [dbo].[_skill_manage] @job = 1, @charid = @DestinationCharID, @param1 = @mastery_id, @param2 = @mastery_level;
    FETCH NEXT FROM cur INTO @mastery_id
END

CLOSE cur
DEALLOCATE cur

-- Add skills for a specific character
DECLARE @skill_id INT

DECLARE @skill_ids TABLE (val INT)
INSERT INTO @skill_ids (val) VALUES (28), (37), (114), (131), (298), (300), (322), (339), (371), (554), (588), (610), (644), (691), (1253), (1256), (1271), (1272), (1281), (1315), (1335), (1343), (1360), (1377), (1380), (1398), (1399), (1410), (1421), (1441), (1449), (1501), (8312), (21209), (30577)

DECLARE cur CURSOR FOR SELECT val FROM @skill_ids
OPEN cur

FETCH NEXT FROM cur INTO @skill_id
WHILE @@FETCH_STATUS = 0
BEGIN
    EXEC [dbo].[_skill_manage] @job = 0, @charid = @DestinationCharID, @param1 = 0, @param2 = @skill_id;
    FETCH NEXT FROM cur INTO @skill_id
END

CLOSE cur
DEALLOCATE cur

GO
