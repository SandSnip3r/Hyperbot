#include "parsing.hpp"

namespace pk2::parsing {

std::string fileDataToString(const std::vector<uint8_t> &data) {
  // TODO: Looks like the text is utf16. be more precise here
	if (data.size()%2 != 0) {
		throw std::runtime_error("Data is not evenly sized");
	}
	std::string result;
	result.reserve((data.size()-2)/2);
	for (int i=2; i<data.size(); i+=2) {
		result += (char)data[i];
	}
	return result;
}

pk2::ref::Skill parseSkilldataLine(const std::string &line) {
  // uint8_t service // 0
  // int32_t id // 1
  // int32_t groupID // 2
  // std::string basic_Code // 3
  // std::string basic_Name // 4
  // std::string basic_Group // 5
  // int32_t basic_Original // 6
  // uint8_t basic_Level // 7
  // uint8_t basic_Activity // 8
  // int32_t basic_ChainCode // 9
  // int32_t basic_RecycleCost // 10
  // int32_t action_PreparingTime // 11
  // int32_t action_CastingTime // 12
  // int32_t action_ActionDuration // 13
  // int32_t action_ReuseDelay // 14
  // int32_t action_CoolTime // 15
  // int32_t action_FlyingSpeed // 16
  // uint8_t action_Interruptable // 17
  // int32_t action_Overlap // 18
  // uint8_t action_AutoAttackType // 19
  // uint8_t action_InTown // 20
  // int16_t action_Range // 21
  // uint8_t target_Required // 22
  // uint8_t targetType_Animal // 23
  // uint8_t targetType_Land // 24
  // uint8_t targetType_Building // 25
  // uint8_t targetGroup_Self // 26
  // uint8_t targetGroup_Ally // 27
  // uint8_t targetGroup_Party // 28
  // uint8_t targetGroup_Enemy_M // 29
  // uint8_t targetGroup_Enemy_P // 30
  // uint8_t targetGroup_Neutral // 31
  // uint8_t targetGroup_DontCare // 32
  // uint8_t targetEtc_SelectDeadBody // 33
  // int32_t reqCommon_Mastery1 // 34
  // int32_t reqCommon_Mastery2 // 35
  // uint8_t reqCommon_MasteryLevel1 // 36
  // uint8_t reqCommon_MasteryLevel2 // 37
  // int16_t reqCommon_Str // 38
  // int16_t reqCommon_Int // 39
  // int32_t reqLearn_Skill1 // 40
  // int32_t reqLearn_Skill2 // 41
  // int32_t reqLearn_Skill3 // 42
  // uint8_t reqLearn_SkillLevel1 // 43
  // uint8_t reqLearn_SkillLevel2 // 44
  // uint8_t reqLearn_SkillLevel3 // 45
  // int32_t reqLearn_SP // 46
  // uint8_t reqLearn_Race // 47
  // uint8_t req_Restriction1 // 48
  // uint8_t req_Restriction2 // 49
  // uint8_t reqCast_Weapon1 // 50
  // uint8_t reqCast_Weapon2 // 51
  // int16_t consume_HP // 52
  // int16_t consume_MP // 53
  // int16_t consume_HPRatio // 54
  // int16_t consume_MPRatio // 55
  // uint8_t consume_WHAN // 56
  // uint8_t uI_SkillTab // 57
  // uint8_t uI_SkillPage // 58
  // uint8_t uI_SkillColumn // 59
  // uint8_t uI_SkillRow // 60
  // std::string uI_IconFile // 61
  // std::string uI_SkillName // 62
  // std::string uI_SkillToolTip // 63
  // std::string uI_SkillToolTip_Desc // 64
  // std::string uI_SkillStudy_Desc // 65
  // int16_t aI_AttackChance // 66
  // uint8_t aI_SkillType // 67
  // int32_t param1 // 68
  // int32_t param2 // 69
  // int32_t param3 // 70
  // int32_t param4 // 71
  // int32_t param5 // 72
  // int32_t param6 // 73
  // int32_t param7 // 74
  // int32_t param8 // 75
  // int32_t param9 // 76
  // int32_t param10 // 77
  // int32_t param11 // 78
  // int32_t param12 // 79
  // int32_t param13 // 80
  // int32_t param14 // 81
  // int32_t param15 // 82
  // int32_t param16 // 83
  // int32_t param17 // 84
  // int32_t param18 // 85
  // int32_t param19 // 86
  // int32_t param20 // 87
  // int32_t param21 // 88
  // int32_t param22 // 89
  // int32_t param23 // 90
  // int32_t param24 // 91
  // int32_t param25 // 92
  // int32_t param26 // 93
  // int32_t param27 // 94
  // int32_t param28 // 95
  // int32_t param29 // 96
  // int32_t param30 // 97
  // int32_t param31 // 98
  // int32_t param32 // 99
  // int32_t param33 // 100
  // int32_t param34 // 101
  // int32_t param35 // 102
  // int32_t param36 // 103
  // int32_t param37 // 104
  // int32_t param38 // 105
  // int32_t param39 // 106
  // int32_t param40 // 107
  // int32_t param41 // 108
  // int32_t param42 // 109
  // int32_t param43 // 110
  // int32_t param44 // 111
  // int32_t param45 // 112
  // int32_t param46 // 113
  // int32_t param47 // 114
  // int32_t param48 // 115
  // int32_t param49 // 116
  // int32_t param50 // 117

	const std::vector<int> kDesiredFields = {1,3,4,5,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing item data, but line contains wrong number of fields");
	}
	pk2::ref::Skill skill;
  int idx=0;
  skill.id =      std::stol(fields[idx++]);
  skill.basicCode =         fields[idx++];
  skill.basicName =         fields[idx++];
  skill.basicGroup =        fields[idx++];
  for (int i=0; i<skill.params.size(); ++i) {
    skill.params[i] = std::stoi(fields[idx++]);
  }
	return skill;
}

pk2::ref::Character parseCharacterdataLine(const std::string &line) {
	// int32_t service; // 0
	// int32_t id; // 1
	// std::string codeName128; // 2
	// std::string objName128; // 3
	// std::string orgObjCodeName128; // 4
	// std::string nameStrID128; // 5
	// std::string descStrID128; // 6
	// uint8_t cashItem; // 7
	// uint8_t bionic; // 8
	// uint8_t typeId1; // 9
	// uint8_t typeId2; // 10
	// uint8_t typeId3; // 11
	// uint8_t typeId4; // 12
	// int32_t decayTime; // 13
	// uint8_t country; // 14
	// uint8_t rarity; // 15
	// uint8_t canTrade; // 16
	// uint8_t canSell; // 17
	// uint8_t canBuy; // 18
	// uint8_t canBorrow; // 19
	// uint8_t canDrop; // 20
	// uint8_t canPick; // 21
	// uint8_t canRepair; // 22
	// uint8_t canRevive; // 23
	// uint8_t canUse; // 24
	// uint8_t canThrow; // 25
	// int32_t price; // 26
	// int32_t costRepair; // 27
	// int32_t costRevive; // 28
	// int32_t costBorrow; // 29
	// int32_t keepingFee; // 30
	// int32_t sellPrice; // 31
	// int32_t reqLevelType1; // 32
	// uint8_t reqLevel1; // 33
	// int32_t reqLevelType2; // 34
	// uint8_t reqLevel2; // 35
	// int32_t reqLevelType3; // 36
	// uint8_t reqLevel3; // 37
	// int32_t reqLevelType4; // 38
	// uint8_t reqLevel4; // 39
	// int32_t maxContain; // 40
	// int16_t regionID; // 41
	// int16_t dir; // 42
	// int16_t offsetX; // 43
	// int16_t offsetY; // 44
	// int16_t offsetZ; // 45
	// int16_t speed1; // 46
	// int16_t speed2; // 47
	// int32_t scale; // 48
	// int16_t bCHeight; // 49
	// int16_t bCRadius; // 50
	// int32_t eventID; // 51
	// std::string assocFileObj128; // 52
	// std::string assocFileDrop128; // 53
	// std::string assocFileIcon128; // 54
	// std::string assocFile1_128; // 55
	// std::string assocFile2_128; // 56
  // uint8_t lvl; // 57
  // uint8_t charGender; // 58
  // int32_t maxHP; // 59
  // int32_t maxMP; // 60
  // int32_t resistFrozen; // 61
  // int32_t resistFrostbite; // 62
  // int32_t resistBurn; // 63
  // int32_t resistEShock; // 64
  // int32_t resistPoison; // 65
  // int32_t resistZombie; // 66
  // int32_t resistSleep; // 67
  // int32_t resistRoot; // 68
  // int32_t resistSlow; // 69
  // int32_t resistFear; // 70
  // int32_t resistMyopia; // 71
  // int32_t resistBlood; // 72
  // int32_t resistStone; // 73
  // int32_t resistDark; // 74
  // int32_t resistStun; // 75
  // int32_t resistDisea; // 76
  // int32_t resistChaos; // 77
  // int32_t resistCsePD; // 78
  // int32_t resistCseMD; // 79
  // int32_t resistCseSTR; // 80
  // int32_t resistCseINT; // 81
  // int32_t resistCseHP; // 82
  // int32_t resistCseMP; // 83
  // int32_t resist24; // 84
  // int32_t resistBomb; // 85
  // int32_t resist26; // 86
  // int32_t resist27; // 87
  // int32_t resist28; // 88
  // int32_t resist29; // 89
  // int32_t resist30; // 90
  // int32_t resist31; // 91
  // int32_t resist32; // 92
  // uint8_t inventorySize; // 93
  // uint8_t canStore_TID1; // 94
  // uint8_t canStore_TID2; // 95
  // uint8_t canStore_TID3; // 96
  // uint8_t canStore_TID4; // 97
  // uint8_t canBeVehicle; // 98
  // uint8_t canControl; // 99
  // uint8_t damagePortion; // 100
  // int16_t maxPassenger; // 101
  // int32_t assocTactics; // 102
  // int32_t pD; // 103
  // int32_t mD; // 104
  // int32_t pAR; // 105
  // int32_t mAR; // 106
  // int32_t eR; // 107
  // int32_t bR; // 108
  // int32_t hR; // 109
  // int32_t cHR; // 110
  // int32_t expToGive; // 111
  // int32_t creepType; // 112
  // uint8_t knockdown; // 113
  // int32_t kO_RecoverTime; // 114
  // int32_t defaultSkill_1; // 115
  // int32_t defaultSkill_2; // 116
  // int32_t defaultSkill_3; // 117
  // int32_t defaultSkill_4; // 118
  // int32_t defaultSkill_5; // 119
  // int32_t defaultSkill_6; // 120
  // int32_t defaultSkill_7; // 121
  // int32_t defaultSkill_8; // 122
  // int32_t defaultSkill_9; // 123
  // int32_t defaultSkill_10; // 124
  // uint8_t textureType; // 125
  // int32_t except_1; // 126
  // int32_t except_2; // 127
  // int32_t except_3; // 128
  // int32_t except_4; // 129
  // int32_t except_5; // 130
  // int32_t except_6; // 131
  // int32_t except_7; // 132
  // int32_t except_8; // 133
  // int32_t except_9; // 134
  // int32_t except_10; // 135
  // int32_t link; // 136

	const std::vector<int> kDesiredFields = {1,2,9,10,11,12,14,58};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing character data, but line contains wrong number of fields");
	}
  int idx=0;
	pk2::ref::Character character;
	character.id =          std::stol(fields[idx++]);
	character.codeName128 =           fields[idx++];
	character.typeId1 =     std::stoi(fields[idx++]);
	character.typeId2 =     std::stoi(fields[idx++]);
	character.typeId3 =     std::stoi(fields[idx++]);
	character.typeId4 =     std::stoi(fields[idx++]);
	character.country =     std::stoi(fields[idx++]);
	character.charGender =  std::stoi(fields[idx++]);
	return character;
}

pk2::ref::Item parseItemdataLine(const std::string &line) {
	// int32_t service; // 0
	// int32_t id; // 1
	// std::string codeName128; // 2
	// std::string objName128; // 3
	// std::string orgObjCodeName128; // 4
	// std::string nameStrID128; // 5
	// std::string descStrID128; // 6
	// uint8_t cashItem; // 7
	// uint8_t bionic; // 8
	// uint8_t typeId1; // 9
	// uint8_t typeId2; // 10
	// uint8_t typeId3; // 11
	// uint8_t typeId4; // 12
	// int32_t decayTime; // 13
	// uint8_t country; // 14
	// uint8_t rarity; // 15
	// uint8_t canTrade; // 16
	// uint8_t canSell; // 17
	// uint8_t canBuy; // 18
	// uint8_t canBorrow; // 19
	// uint8_t canDrop; // 20
	// uint8_t canPick; // 21
	// uint8_t canRepair; // 22
	// uint8_t canRevive; // 23
	// uint8_t canUse; // 24
	// uint8_t canThrow; // 25
	// int32_t price; // 26
	// int32_t costRepair; // 27
	// int32_t costRevive; // 28
	// int32_t costBorrow; // 29
	// int32_t keepingFee; // 30
	// int32_t sellPrice; // 31
	// int32_t reqLevelType1; // 32
	// uint8_t reqLevel1; // 33
	// int32_t reqLevelType2; // 34
	// uint8_t reqLevel2; // 35
	// int32_t reqLevelType3; // 36
	// uint8_t reqLevel3; // 37
	// int32_t reqLevelType4; // 38
	// uint8_t reqLevel4; // 39
	// int32_t maxContain; // 40
	// int16_t regionID; // 41
	// int16_t dir; // 42
	// int16_t offsetX; // 43
	// int16_t offsetY; // 44
	// int16_t offsetZ; // 45
	// int16_t speed1; // 46
	// int16_t speed2; // 47
	// int32_t scale; // 48
	// int16_t bCHeight; // 49
	// int16_t bCRadius; // 50
	// int32_t eventID; // 51
	// std::string assocFileObj128; // 52
	// std::string assocFileDrop128; // 53
	// std::string assocFileIcon128; // 54
	// std::string assocFile1_128; // 55
	// std::string assocFile2_128; // 56
	// int32_t maxStack; // 57
	// uint8_t reqGender; // 58
	// int32_t reqStr; // 59
	// int32_t reqInt; // 60
	// uint8_t itemClass; // 61
	// int32_t setID; // 62
	// float dur_L; // 63
	// float dur_U; // 64
	// float pD_L; // 65
	// float pD_U; // 66
	// float pDInc; // 67
	// float eR_L; // 68
	// float eR_U; // 69
	// float eRInc; // 70
	// float pAR_L; // 71
	// float pAR_U; // 72
	// float pARInc; // 73
	// float bR_L; // 74
	// float bR_U; // 75
	// float mD_L; // 76
	// float mD_U; // 77
	// float mDInc; // 78
	// float mAR_L; // 79
	// float mAR_U; // 80
	// float mARInc; // 81
	// float pDStr_L; // 82
	// float pDStr_U; // 83
	// float mDInt_L; // 84
	// float mDInt_U; // 85
	// uint8_t quivered; // 86
	// uint8_t ammo1_TID4; // 87
	// uint8_t ammo2_TID4; // 88
	// uint8_t ammo3_TID4; // 89
	// uint8_t ammo4_TID4; // 90
	// uint8_t ammo5_TID4; // 91
	// uint8_t speedClass; // 92
	// uint8_t twoHanded; // 93
	// int16_t range; // 94
	// float pAttackMin_L; // 95
	// float pAttackMin_U; // 96
	// float pAttackMax_L; // 97
	// float pAttackMax_U; // 98
	// float pAttackInc; // 99
	// float mAttackMin_L; // 100
	// float mAttackMin_U; // 101
	// float mAttackMax_L; // 102
	// float mAttackMax_U; // 103
	// float mAttackInc; // 104
	// float pAStrMin_L; // 105
	// float pAStrMin_U; // 106
	// float pAStrMax_L; // 107
	// float pAStrMax_U; // 108
	// float mAInt_Min_L; // 109
	// float mAInt_Min_U; // 110
	// float mAInt_Max_L; // 111
	// float mAInt_Max_U; // 112
	// float hR_L; // 113
	// float hR_U; // 114
	// float hRInc; // 115
	// float cHR_L; // 116
	// float cHR_U; // 117
	// int32_t param1; // 118
	// std::string desc1_128; // 119
	// int32_t param2; // 120
	// char desc2_128; // 121
	// int32_t param3; // 122
	// std::string desc3_128; // 123
	// int32_t param4; // 124
	// std::string desc4_128; // 125
	// int32_t param5; // 126
	// std::string desc5_128; // 127
	// int32_t param6; // 128
	// std::string desc6_128; // 129
	// int32_t param7; // 130
	// std::string desc7_128; // 131
	// int32_t param8; // 132
	// std::string desc8_128; // 133
	// int32_t param9; // 134
	// std::string desc9_128; // 135
	// int32_t param10; // 136
	// std::string desc10_128; // 137
	// int32_t param11; // 138
	// std::string desc11_128; // 139
	// int32_t param12; // 140
	// std::string desc12_128; // 141
	// int32_t param13; // 142
	// std::string desc13_128; // 143
	// int32_t param14; // 144
	// std::string desc14_128; // 145
	// int32_t param15; // 146
	// std::string desc15_128; // 147
	// int32_t param16; // 148
	// std::string desc16_128; // 149
	// int32_t param17; // 150
	// std::string desc17_128; // 151
	// int32_t param18; // 152
	// std::string desc18_128; // 153
	// int32_t param19; // 154
	// std::string desc19_128; // 155
	// int32_t param20; // 156
	// std::string desc20_128; // 157
	// uint8_t maxMagicOptCount; // 158
	// uint8_t childItemCount; // 159
  
	const std::vector<int> kDesiredFields = {1,2,7,8,9,10,11,12,57,118,120,124};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing item data, but line contains wrong number of fields");
	}
	pk2::ref::Item item;
  int idx=0;
	item.id =       std::stol(fields[idx++]);
	item.codeName128 =        fields[idx++];
	item.cashItem = std::stoi(fields[idx++]);
	item.bionic =   std::stoi(fields[idx++]);
	item.typeId1 =  std::stoi(fields[idx++]);
	item.typeId2 =  std::stoi(fields[idx++]);
	item.typeId3 =  std::stoi(fields[idx++]);
	item.typeId4 =  std::stoi(fields[idx++]);
	item.maxStack = std::stoi(fields[idx++]);
	item.param1 =   std::stol(fields[idx++]);
	item.param2 =   std::stol(fields[idx++]);
	item.param4 =   std::stol(fields[idx++]);
	return item;
}

pk2::ref::Teleport parseTeleportbuildingLine(const std::string &line) {
  // int32_t service // 0
  // int32_t id // 1
  // std::string codeName128 // 2
  // std::string objName128 // 3
  // std::string orgObjCodeName128 // 4
  // std::string nameStrID128 // 5
  // std::string descStrID128 // 6
  // uint8_t cashItem // 7
  // uint8_t bionic // 8
  // uint8_t typeId1 // 9
  // uint8_t typeId2 // 10
  // uint8_t typeId3 // 11
  // uint8_t typeId4 // 12
  // int32_t decayTime // 13
  // uint8_t country // 14
  // uint8_t rarity // 15
  // uint8_t canTrade // 16
  // uint8_t canSell // 17
  // uint8_t canBuy // 18
  // uint8_t canBorrow // 19
  // uint8_t canDrop // 20
  // uint8_t canPick // 21
  // uint8_t canRepair // 22
  // uint8_t canRevive // 23
  // uint8_t canUse // 24
  // uint8_t canThrow // 25
  // int32_t price // 26
  // int32_t costRepair // 27
  // int32_t costRevive // 28
  // int32_t costBorrow // 29
  // int32_t keepingFee // 30
  // int32_t sellPrice // 31
  // int32_t reqLevelType1 // 32
  // uint8_t reqLevel1 // 33
  // int32_t reqLevelType2 // 34
  // uint8_t reqLevel2 // 35
  // int32_t reqLevelType3 // 36
  // uint8_t reqLevel3 // 37
  // int32_t reqLevelType4 // 38
  // uint8_t reqLevel4 // 39
  // int32_t maxContain // 40
  // int16_t regionID // 41
  // int16_t dir // 42
  // int16_t offsetX // 43
  // int16_t offsetY // 44
  // int16_t offsetZ // 45
  // int16_t speed1 // 46
  // int16_t speed2 // 47
  // int32_t scale // 48
  // int16_t bcHeight // 49
  // int16_t bcRadius // 50
  // int32_t eventID // 51
  // std::string assocFileObj128 // 52
  // std::string assocFileDrop128 // 53
  // std::string assocFileIcon128 // 54
  // std::string assocFile1_128 // 55
  // std::string assocFile2_128 // 56
  // int32_t link // 57
	
	const std::vector<int> kDesiredFields = {1,2,9,10,11,12};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing teleport data, but line contains wrong number of fields");
	}
	pk2::ref::Teleport teleport;
  int idx=0;
	teleport.id =       std::stol(fields[idx++]);
	teleport.codeName128 =        fields[idx++];
	teleport.typeId1 =  std::stoi(fields[idx++]);
	teleport.typeId2 =  std::stoi(fields[idx++]);
	teleport.typeId3 =  std::stoi(fields[idx++]);
	teleport.typeId4 =  std::stoi(fields[idx++]);
	return teleport;
}

pk2::ref::ScrapOfPackageItem parseScrapOfPackageItemLine(const std::string &line) {
  // uint8_t service // 0
  // int32_t country // 1
  // std::string refPackageItemCodeName // 2
  // std::string refItemCodeName // 3
  // uint8_t optLevel // 4
  // int64_t variance // 5
  // int32_t data // 6
  // uint8_t magParamNum // 7
  // int64_t magParam1 // 8
  // int64_t magParam2 // 9
  // int64_t magParam3 // 10
  // int64_t magParam4 // 11
  // int64_t magParam5 // 12
  // int64_t magParam6 // 13
  // int64_t magParam7 // 14
  // int64_t magParam8 // 15
  // int64_t magParam9 // 16
  // int64_t magParam10 // 17
  // int64_t magParam11 // 18
  // int64_t magParam12 // 19
  // int32_t param1 // 20
  // std::string param1_Desc128 // 21
  // int32_t param2 // 22
  // std::string param2_Desc128 // 23
  // int32_t param3 // 24
  // std::string param3_Desc128 // 25
  // int32_t param4 // 26
  // std::string param4_Desc128 // 27
  // int32_t index // 28
	const std::vector<int> kDesiredFields = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing scrap of package item, but line contains wrong number of fields");
	}
	pk2::ref::ScrapOfPackageItem scrap;
  int idx=0;
	scrap.refPackageItemCodeName = fields[idx++];
	scrap.refItemCodeName =        fields[idx++];
	scrap.optLevel =     std::stoi(fields[idx++]);
	scrap.variance =    std::stoll(fields[idx++]);
	scrap.data =         std::stol(fields[idx++]);
  scrap.magParamNum =  std::stoi(fields[idx++]);
  scrap.magParams[0] =   std::stoll(fields[idx++]);
  scrap.magParams[1] =   std::stoll(fields[idx++]);
  scrap.magParams[2] =   std::stoll(fields[idx++]);
  scrap.magParams[3] =   std::stoll(fields[idx++]);
  scrap.magParams[4] =   std::stoll(fields[idx++]);
  scrap.magParams[5] =   std::stoll(fields[idx++]);
  scrap.magParams[6] =   std::stoll(fields[idx++]);
  scrap.magParams[7] =   std::stoll(fields[idx++]);
  scrap.magParams[8] =   std::stoll(fields[idx++]);
  scrap.magParams[9] =  std::stoll(fields[idx++]);
  scrap.magParams[10] =  std::stoll(fields[idx++]);
  scrap.magParams[11] =  std::stoll(fields[idx++]);
	return scrap;
}

pk2::ref::ShopTab parseShopTabLine(const std::string &line) {
  // uint8_t service // 0
  // int32_t country // 1
  // int32_t id // 2
  // std::string codeName128 // 3
  // std::string refTabGroupCodeName // 4
  // std::string strID128_Tab // 5
	const std::vector<int> kDesiredFields = {3,4};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing shop tab, but line contains wrong number of fields");
	}
	pk2::ref::ShopTab tab;
  int idx=0;
	tab.codeName128 =         fields[idx++];
	tab.refTabGroupCodeName = fields[idx++];
	return tab;
}

pk2::ref::ShopGroup parseShopGroupLine(const std::string &line) {
  // uint8_t service // 0
  // int32_t country // 1
  // in16_t id // 2
  // std::string codeName128 // 3
  // std::string refNPCCodeName // 4
  // int32_t param1 // 5
  // std::string param1_Desc128 // 6
  // int32_t param2 // 7
  // std::string param2_Desc128 // 8
  // int32_t param3 // 9
  // std::string param3_Desc128 // 10
  // int32_t param4 // 11
  // std::string param4_Desc128 // 12
	const std::vector<int> kDesiredFields = {3,4};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing shop group, but line contains wrong number of fields");
	}
	pk2::ref::ShopGroup tab;
  int idx=0;
	tab.codeName128 =    fields[idx++];
	tab.refNPCCodeName = fields[idx++];
	return tab;
}

pk2::ref::ShopGood parseShopGoodLine(const std::string &line) {
  // uint8_t service // 0
  // int32_t country // 1
  // std::string refTabCodeName // 2
  // std::string refPackageItemCodeName // 3
  // uint8_t slotIndex // 4
  // int32_t param1 // 5
  // std::string param1_Desc128 // 6
  // int32_t param2 // 7
  // std::string param2_Desc128 // 8
  // int32_t param3 // 9
  // std::string param3_Desc128 // 10
  // int32_t param4 // 11
  // std::string param4_Desc128 // 12
	const std::vector<int> kDesiredFields = {2,3,4};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing shop good, but line contains wrong number of fields");
	}
	pk2::ref::ShopGood good;
  int idx=0;
	good.refTabCodeName =         fields[idx++];
	good.refPackageItemCodeName = fields[idx++];
	good.slotIndex =    std::stoi(fields[idx++]);
	return good;
}

pk2::ref::MappingShopGroup parseMappingShopGroupLine(const std::string &line) {
  // uint8_t service // 0
  // int32_t country // 1
  // std::string refShopGroupCodeName // 2
  // std::string refShopCodeName // 3
	const std::vector<int> kDesiredFields = {2,3};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing mapping shop group, but line contains wrong number of fields");
	}
	pk2::ref::MappingShopGroup mapping;
  int idx=0;
	mapping.refShopGroupCodeName = fields[idx++];
	mapping.refShopCodeName =      fields[idx++];
	return mapping;
}

pk2::ref::MappingShopWithTab parseMappingShopWithTabLine(const std::string &line) {
  // uint8_t service // 3
  // int32_t country // 2
  // std::string refShopCodeName // 1
  // std::string refTabGroupCodeName // 0
	const std::vector<int> kDesiredFields = {2,3};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing mapping shop with tab, but line contains wrong number of fields");
	}
	pk2::ref::MappingShopWithTab mapping;
  int idx=0;
	mapping.refShopCodeName =     fields[idx++];
	mapping.refTabGroupCodeName = fields[idx++];
	return mapping;
}

DivisionInfo parseDivisionInfo(const std::vector<uint8_t> &data) {
	DivisionInfo divisionInfo;
	int readIndex=0;
	divisionInfo.locale = get<uint8_t>(data, readIndex);
	auto divisionCount = get<uint8_t>(data, readIndex);
	for (int i=0; i<divisionCount; ++i) {
		Division division;
		auto divisionNameLength = get<uint32_t>(data, readIndex);
		for (int j=0; j<divisionNameLength; ++j) {
			char c = get<char>(data, readIndex);
			division.name += c;
		}
		++readIndex; // throw away null terminator
		auto gatewayCount = get<uint8_t>(data, readIndex);
		for (int j=0; j<gatewayCount; ++j) {
			auto addressLength = get<uint32_t>(data, readIndex);
			std::string address;
			for (int k=0; k<addressLength; ++k) {
				char c = get<char>(data, readIndex);
				address += c;
			}
			++readIndex; // throw away null terminator
			division.gatewayIpAddresses.push_back(address);
			divisionInfo.divisions.push_back(division);
		}
	}
	return divisionInfo;
}

std::vector<std::string> split(const std::string &str, const std::string &delim) {
	std::vector<std::string> result;
	size_t last = 0;
	size_t next = 0;
	while ((next = str.find(delim, last)) != std::string::npos) {
		std::string s = str.substr(last, next-last);
		if (s != "") {
			result.push_back(s);
		}
		last = next + delim.size();
	}
	std::string s = str.substr(last);
	if (s != "") {
		result.push_back(s);
	}
	return result;		
}

std::vector<std::string> splitAndSelectFields(const std::string &str, const std::string &delim, const std::vector<int> &fields) {
	if (fields.empty()) {
		return {};
	}
	int fieldsIndex=0;
	std::vector<std::string> result;
	size_t last = 0;
	size_t next = 0;
	int fieldNum=0;
	while ((next = str.find(delim, last)) != std::string::npos) {
		std::string s = str.substr(last, next-last);
		if (s != "") {
			if (fields[fieldsIndex] == fieldNum) {
				result.push_back(s);
				++fieldsIndex;
				if (fieldsIndex >= fields.size()) {
					return result;
				}
			}
		}
		last = next + delim.size();
		++fieldNum;
	}
	std::string s = str.substr(last);
	if (s != "") {
		if (fields[fieldsIndex] == fieldNum) {
			result.push_back(s);
		}
	}
	return result;		
}

} // namespace pk2::parsing