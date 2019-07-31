#include "parsing.hpp"
#include "item.hpp"

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

pk2::media::Item parseItemdataLine(const std::string &line) {
	// int service; //0
	// int id; //1
	// std::string codeName128; //2
	// std::string objName128; //3
	// std::string orgObjCodeName128; //4
	// std::string nameStrID128; //5
	// std::string descStrID128; //6
	// uint8_t cashItem; //7
	// uint8_t bionic; //8
	// uint8_t typeId1; //9
	// uint8_t typeId2; //10
	// uint8_t typeId3; //11
	// uint8_t typeId4; //12
	// int decayTime; //13
	// uint8_t country; //14
	// uint8_t rarity; //15
	// uint8_t canTrade; //16
	// uint8_t canSell; //17
	// uint8_t canBuy; //18
	// uint8_t canBorrow; //19
	// uint8_t canDrop; //20
	// uint8_t canPick; //21
	// uint8_t canRepair; //22
	// uint8_t canRevive; //23
	// uint8_t canUse; //24
	// uint8_t canThrow; //25
	// int price; //26
	// int costRepair; //27
	// int costRevive; //28
	// int costBorrow; //29
	// int keepingFee; //30
	// int sellPrice; //31
	// int reqLevelType1; //32
	// uint8_t reqLevel1; //33
	// int reqLevelType2; //34
	// uint8_t reqLevel2; //35
	// int reqLevelType3; //36
	// uint8_t reqLevel3; //37
	// int reqLevelType4; //38
	// uint8_t reqLevel4; //39
	// int maxContain; //40
	// int16_t regionID; //41
	// int16_t dir; //42
	// int16_t offsetX; //43
	// int16_t offsetY; //44
	// int16_t offsetZ; //45
	// int16_t speed1; //46
	// int16_t speed2; //47
	// int scale; //48
	// int16_t bCHeight; //49
	// int16_t bCRadius; //50
	// int eventID; //51
	// std::string assocFileObj128; //52
	// std::string assocFileDrop128; //53
	// std::string assocFileIcon128; //54
	// std::string assocFile1_128; //55
	// std::string assocFile2_128; //56
	// int maxStack; //57
	// uint8_t reqGender; //58
	// int reqStr; //59
	// int reqInt; //60
	// uint8_t itemClass; //61
	// int setID; //62
	// float dur_L; //63
	// float dur_U; //64
	// float pD_L; //65
	// float pD_U; //66
	// float pDInc; //67
	// float eR_L; //68
	// float eR_U; //69
	// float eRInc; //70
	// float pAR_L; //71
	// float pAR_U; //72
	// float pARInc; //73
	// float bR_L; //74
	// float bR_U; //75
	// float mD_L; //76
	// float mD_U; //77
	// float mDInc; //78
	// float mAR_L; //79
	// float mAR_U; //80
	// float mARInc; //81
	// float pDStr_L; //82
	// float pDStr_U; //83
	// float mDInt_L; //84
	// float mDInt_U; //85
	// uint8_t quivered; //86
	// uint8_t ammo1_TID4; //87
	// uint8_t ammo2_TID4; //88
	// uint8_t ammo3_TID4; //89
	// uint8_t ammo4_TID4; //90
	// uint8_t ammo5_TID4; //91
	// uint8_t speedClass; //92
	// uint8_t twoHanded; //93
	// int16_t range; //94
	// float pAttackMin_L; //95
	// float pAttackMin_U; //96
	// float pAttackMax_L; //97
	// float pAttackMax_U; //98
	// float pAttackInc; //99
	// float mAttackMin_L; //100
	// float mAttackMin_U; //101
	// float mAttackMax_L; //102
	// float mAttackMax_U; //103
	// float mAttackInc; //104
	// float pAStrMin_L; //105
	// float pAStrMin_U; //106
	// float pAStrMax_L; //107
	// float pAStrMax_U; //108
	// float mAInt_Min_L; //109
	// float mAInt_Min_U; //110
	// float mAInt_Max_L; //111
	// float mAInt_Max_U; //112
	// float hR_L; //113
	// float hR_U; //114
	// float hRInc; //115
	// float cHR_L; //116
	// float cHR_U; //117
	// int param1; //118
	// std::string desc1_128; //119
	// int param2; //120
	// char desc2_128; //121
	// int param3; //122
	// std::string desc3_128; //123
	// int param4; //124
	// std::string desc4_128; //125
	// int param5; //126
	// std::string desc5_128; //127
	// int param6; //128
	// std::string desc6_128; //129
	// int param7; //130
	// std::string desc7_128; //131
	// int param8; //132
	// std::string desc8_128; //133
	// int param9; //134
	// std::string desc9_128; //135
	// int param10; //136
	// std::string desc10_128; //137
	// int param11; //138
	// std::string desc11_128; //139
	// int param12; //140
	// std::string desc12_128; //141
	// int param13; //142
	// std::string desc13_128; //143
	// int param14; //144
	// std::string desc14_128; //145
	// int param15; //146
	// std::string desc15_128; //147
	// int param16; //148
	// std::string desc16_128; //149
	// int param17; //150
	// std::string desc17_128; //151
	// int param18; //152
	// std::string desc18_128; //153
	// int param19; //154
	// std::string desc19_128; //155
	// int param20; //156
	// std::string desc20_128; //157
	// uint8_t maxMagicOptCount; //158
	// uint8_t childItemCount; //159
	// auto fields = split(line, "\t");
	const std::vector<int> kDesiredFields = {1,2,9,10,11,12};
	auto fields = splitAndSelectFields(line, "\t", kDesiredFields);
	if (fields.size() != kDesiredFields.size()) {
		// TODO: This check for validity of data should be more robust
		throw std::runtime_error("Parsing item data, but line contains wrong number of fields");
	}
	pk2::media::Item item;
	item.id = std::stoi(fields[0]);
	item.codeName128 = fields[1];
	item.typeId1 = std::stoi(fields[2]);
	item.typeId2 = std::stoi(fields[3]);
	item.typeId3 = std::stoi(fields[4]);
	item.typeId4 = std::stoi(fields[5]);
	return item;
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