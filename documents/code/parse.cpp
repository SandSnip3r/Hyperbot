#include <codecvt>
#include <iostream>
#include <fstream>
#include <locale>
#include <string>
#include <string_view>
#include <vector>

using namespace std;

struct ParsedLine {
  int32_t var0; // Service	int
  int32_t var1; // ID	int
  string var2; // CodeName128	varchar(129)
  string var3; // ObjName128	varchar(129)
  string var4; // OrgObjCodeName128	varchar(129)
  string var5; // NameStrID128	varchar(129)
  string var6; // DescStrID128	varchar(129)
  uint8_t var7; // CashItem	tinyint
  uint8_t var8; // Bionic	tinyint
  uint8_t var9; // TypeID1	tinyint
  uint8_t var10; // TypeID2	tinyint
  uint8_t var11; // TypeID3	tinyint
  uint8_t var12; // TypeID4	tinyint
  int32_t var13; // DecayTime	int
  uint8_t var14; // Country	tinyint
  uint8_t var15; // Rarity	tinyint
  uint8_t var16; // CanTrade	tinyint
  uint8_t var17; // CanSell	tinyint
  uint8_t var18; // CanBuy	tinyint
  uint8_t var19; // CanBorrow	tinyint
  uint8_t var20; // CanDrop	tinyint
  uint8_t var21; // CanPick	tinyint
  uint8_t var22; // CanRepair	tinyint
  uint8_t var23; // CanRevive	tinyint
  uint8_t var24; // CanUse	tinyint
  uint8_t var25; // CanThrow	tinyint
  int32_t var26; // Price	int
  int32_t var27; // CostRepair	int
  int32_t var28; // CostRevive	int
  int32_t var29; // CostBorrow	int
  int32_t var30; // KeepingFee	int
  int32_t var31; // SellPrice	int
  int32_t var32; // ReqLevelType1	int
  uint8_t var33; // ReqLevel1	tinyint
  int32_t var34; // ReqLevelType2	int
  uint8_t var35; // ReqLevel2	tinyint
  int32_t var36; // ReqLevelType3	int
  uint8_t var37; // ReqLevel3	tinyint
  int32_t var38; // ReqLevelType4	int
  uint8_t var39; // ReqLevel4	tinyint
  int32_t var40; // MaxContain	int
  int16_t var41; // RegionID	smallint
  int16_t var42; // Dir	smallint
  int16_t var43; // OffsetX	smallint
  int16_t var44; // OffsetY	smallint
  int16_t var45; // OffsetZ	smallint
  int16_t var46; // Speed1	smallint
  int16_t var47; // Speed2	smallint
  int32_t var48; // Scale	int
  int16_t var49; // BCHeight	smallint
  int16_t var50; // BCRadius	smallint
  int32_t var51; // EventID	int
  string var52; // AssocFileObj128	varchar(129)
  string var53; // AssocFileDrop128	varchar(129)
  string var54; // AssocFileIcon128	varchar(129)
  string var55; // AssocFile1_128	varchar(129)
  string var56; // AssocFile2_128	varchar(129)
  int32_t var57; // MaxStack	int
  uint8_t var58; // ReqGender	tinyint
  int32_t var59; // ReqStr	int
  int32_t var60; // ReqInt	int
  uint8_t var61; // ItemClass	tinyint
  int32_t var62; // SetID	int
  float var63; // Dur_L	real
  float var64; // Dur_U	real
  float var65; // PD_L	real
  float var66; // PD_U	real
  float var67; // PDInc	real
  float var68; // ER_L	real
  float var69; // ER_U	real
  float var70; // ERInc	real
  float var71; // PAR_L	real
  float var72; // PAR_U	real
  float var73; // PARInc	real
  float var74; // BR_L	real
  float var75; // BR_U	real
  float var76; // MD_L	real
  float var77; // MD_U	real
  float var78; // MDInc	real
  float var79; // MAR_L	real
  float var80; // MAR_U	real
  float var81; // MARInc	real
  float var82; // PDStr_L	real
  float var83; // PDStr_U	real
  float var84; // MDInt_L	real
  float var85; // MDInt_U	real
  uint8_t var86; // Quivered	tinyint
  uint8_t var87; // Ammo1_TID4	tinyint
  uint8_t var88; // Ammo2_TID4	tinyint
  uint8_t var89; // Ammo3_TID4	tinyint
  uint8_t var90; // Ammo4_TID4	tinyint
  uint8_t var91; // Ammo5_TID4	tinyint
  uint8_t var92; // SpeedClass	tinyint
  uint8_t var93; // TwoHanded	tinyint
  int16_t var94; // Range	smallint
  float var95; // PAttackMin_L	real
  float var96; // PAttackMin_U	real
  float var97; // PAttackMax_L	real
  float var98; // PAttackMax_U	real
  float var99; // PAttackInc	real
  float var100; // MAttackMin_L	real
  float var101; // MAttackMin_U	real
  float var102; // MAttackMax_L	real
  float var103; // MAttackMax_U	real
  float var104; // MAttackInc	real
  float var105; // PAStrMin_L	real
  float var106; // PAStrMin_U	real
  float var107; // PAStrMax_L	real
  float var108; // PAStrMax_U	real
  float var109; // MAInt_Min_L	real
  float var110; // MAInt_Min_U	real
  float var111; // MAInt_Max_L	real
  float var112; // MAInt_Max_U	real
  float var113; // HR_L	real
  float var114; // HR_U	real
  float var115; // HRInc	real
  float var116; // CHR_L	real
  float var117; // CHR_U	real
  int32_t var118; // Param1	int
  string var119; // Desc1_128	varchar(129)
  int32_t var120; // Param2	int
  string var121; // Desc2_128	char(129)
  int32_t var122; // Param3	int
  string var123; // Desc3_128	varchar(129)
  int32_t var124; // Param4	int
  string var125; // Desc4_128	varchar(129)
  int32_t var126; // Param5	int
  string var127; // Desc5_128	varchar(129)
  int32_t var128; // Param6	int
  string var129; // Desc6_128	varchar(129)
  int32_t var130; // Param7	int
  string var131; // Desc7_128	varchar(129)
  int32_t var132; // Param8	int
  string var133; // Desc8_128	varchar(129)
  int32_t var134; // Param9	int
  string var135; // Desc9_128	varchar(129)
  int32_t var136; // Param10	int
  string var137; // Desc10_128	varchar(129)
  int32_t var138; // Param11	int
  string var139; // Desc11_128	varchar(129)
  int32_t var140; // Param12	int
  string var141; // Desc12_128	varchar(129)
  int32_t var142; // Param13	int
  string var143; // Desc13_128	varchar(129)
  int32_t var144; // Param14	int
  string var145; // Desc14_128	varchar(129)
  int32_t var146; // Param15	int
  string var147; // Desc15_128	varchar(129)
  int32_t var148; // Param16	int
  string var149; // Desc16_128	varchar(129)
  int32_t var150; // Param17	int
  string var151; // Desc17_128	varchar(129)
  int32_t var152; // Param18	int
  string var153; // Desc18_128	varchar(129)
  int32_t var154; // Param19	int
  string var155; // Desc19_128	varchar(129)
  int32_t var156; // Param20	int
  string var157; // Desc20_128	varchar(129)
  uint8_t var158; // MaxMagicOptCount	tinyint
  uint8_t var159; // ChildItemCount	tinyint
  int32_t var160; // Link	int
};

void copySubstring(const char *begin, char **end, std::string &dest) {
  const char *ptr = begin;
  while (*ptr != 0 && *ptr != '\t') {
    ++ptr;
  }
  string_view sv(begin, ptr-begin);
  dest = sv;
  *end = const_cast<char*>(ptr);
}

template<typename T>
const char* parse(const char *begin, T &result);

template<>
const char* parse<std::string>(const char *begin, std::string &result) {
  char *end;
  copySubstring(begin, &end, result);
  return end+1;
}

template<>
const char* parse<uint8_t>(const char *begin, uint8_t &result) {
  char *end;
  result = strtol(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<int16_t>(const char *begin, int16_t &result) {
  char *end;
  result = strtol(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<int32_t>(const char *begin, int32_t &result) {
  char *end;
  result = strtol(begin, &end, 10);
  return end+1;
}

template<>
const char* parse<float>(const char *begin, float &result) {
  char *end;
  result = strtof(begin, &end);
  return end+1;
}

ParsedLine parseLine(const std::string &line);

void print(const ParsedLine &info);

void adjustLine(std::string &line) {
  if (line.size()%2 != 0) {
    throw std::runtime_error("Line length was expected to be even!");
  }
  for (int i=1; i*2<line.size(); ++i) {
    line[i] = line[i*2];
  }
  line.resize(line.size()/2);
}

bool checkZeroes(const string &str) {
  for (int i=1; i<str.size(); i+=2) {
    if (str[i] != 0) {
      return false;
      cout << "Fuck!\n";
    }
  }
  return true;
}

std::string readEntireFileIntoString(const std::string &filename) {
  std::ifstream fin(filename, std::ios::binary);
  if (!fin) {
    throw std::runtime_error("Failed to open file \""+filename+"\"");
  }
  fin.seekg(0, ios::end);
  size_t size = (size_t)fin.tellg();

  //skip BOM
  fin.seekg(2, ios::beg);
  size -= 2;

  std::u16string u16((size / 2) + 1, '\0');
  fin.read((char*)&u16[0], size);

  return std::wstring_convert<
      std::codecvt_utf8_utf16<char16_t>, char16_t>{}.to_bytes(u16);
}

bool isValidItemDataLine(const std::string &line) {
  if (line.size() < 161+160) {
    return false;
  }
  if (line[0] == '/' && line[1] == '/') {
    return false;
  }
  return true;
}

int main() {
  const std::string kTextDir = "Media/server_dep/silkroad/textdata/";
  const std::vector<std::string> filenames = { "itemdata_5000.txt", "itemdata_10000.txt", "itemdata_15000.txt", "itemdata_20000.txt", "itemdata_25000.txt", "itemdata_30000.txt", "itemdata_35000.txt", "itemdata_40000.txt", "itemdata_45000.txt", "itemdata_50000.txt" };
  std::vector<ParsedLine> itemData;
  for (const auto &filename : filenames) {
    cout << "filename: " << filename << '\n';
    const auto fileContents = readEntireFileIntoString(kTextDir+filename);
    size_t start=0;
    size_t posOfNewline = fileContents.find('\n');
    while ((posOfNewline = fileContents.find('\n', start)) != std::string::npos) {
      const auto line = fileContents.substr(start,posOfNewline-start);
      if (isValidItemDataLine(line)) {
        itemData.emplace_back(parseLine(line));
      }
      start = posOfNewline+1;
    }
    if (start < fileContents.size()-1) {
      // File doesnt end in newline. One more line to read
      cout << "One more\n";
      const auto line = fileContents.substr(start);
      if (isValidItemDataLine(line)) {
        itemData.emplace_back(parseLine(line));
      }
    }
  }
  for (const auto &i : itemData) {
    cout << i.var2 << ' ';
  }
  cout << '\n';
  return 0;
}

ParsedLine parseLine(const std::string &line) {
  ParsedLine result;
  const char *ptr = line.data();
  ptr = parse(ptr, result.var0);
  ptr = parse(ptr, result.var1);
  ptr = parse(ptr, result.var2);
  ptr = parse(ptr, result.var3);
  ptr = parse(ptr, result.var4);
  ptr = parse(ptr, result.var5);
  ptr = parse(ptr, result.var6);
  ptr = parse(ptr, result.var7);
  ptr = parse(ptr, result.var8);
  ptr = parse(ptr, result.var9);
  ptr = parse(ptr, result.var10);
  ptr = parse(ptr, result.var11);
  ptr = parse(ptr, result.var12);
  ptr = parse(ptr, result.var13);
  ptr = parse(ptr, result.var14);
  ptr = parse(ptr, result.var15);
  ptr = parse(ptr, result.var16);
  ptr = parse(ptr, result.var17);
  ptr = parse(ptr, result.var18);
  ptr = parse(ptr, result.var19);
  ptr = parse(ptr, result.var20);
  ptr = parse(ptr, result.var21);
  ptr = parse(ptr, result.var22);
  ptr = parse(ptr, result.var23);
  ptr = parse(ptr, result.var24);
  ptr = parse(ptr, result.var25);
  ptr = parse(ptr, result.var26);
  ptr = parse(ptr, result.var27);
  ptr = parse(ptr, result.var28);
  ptr = parse(ptr, result.var29);
  ptr = parse(ptr, result.var30);
  ptr = parse(ptr, result.var31);
  ptr = parse(ptr, result.var32);
  ptr = parse(ptr, result.var33);
  ptr = parse(ptr, result.var34);
  ptr = parse(ptr, result.var35);
  ptr = parse(ptr, result.var36);
  ptr = parse(ptr, result.var37);
  ptr = parse(ptr, result.var38);
  ptr = parse(ptr, result.var39);
  ptr = parse(ptr, result.var40);
  ptr = parse(ptr, result.var41);
  ptr = parse(ptr, result.var42);
  ptr = parse(ptr, result.var43);
  ptr = parse(ptr, result.var44);
  ptr = parse(ptr, result.var45);
  ptr = parse(ptr, result.var46);
  ptr = parse(ptr, result.var47);
  ptr = parse(ptr, result.var48);
  ptr = parse(ptr, result.var49);
  ptr = parse(ptr, result.var50);
  ptr = parse(ptr, result.var51);
  ptr = parse(ptr, result.var52);
  ptr = parse(ptr, result.var53);
  ptr = parse(ptr, result.var54);
  ptr = parse(ptr, result.var55);
  ptr = parse(ptr, result.var56);
  ptr = parse(ptr, result.var57);
  ptr = parse(ptr, result.var58);
  ptr = parse(ptr, result.var59);
  ptr = parse(ptr, result.var60);
  ptr = parse(ptr, result.var61);
  ptr = parse(ptr, result.var62);
  ptr = parse(ptr, result.var63);
  ptr = parse(ptr, result.var64);
  ptr = parse(ptr, result.var65);
  ptr = parse(ptr, result.var66);
  ptr = parse(ptr, result.var67);
  ptr = parse(ptr, result.var68);
  ptr = parse(ptr, result.var69);
  ptr = parse(ptr, result.var70);
  ptr = parse(ptr, result.var71);
  ptr = parse(ptr, result.var72);
  ptr = parse(ptr, result.var73);
  ptr = parse(ptr, result.var74);
  ptr = parse(ptr, result.var75);
  ptr = parse(ptr, result.var76);
  ptr = parse(ptr, result.var77);
  ptr = parse(ptr, result.var78);
  ptr = parse(ptr, result.var79);
  ptr = parse(ptr, result.var80);
  ptr = parse(ptr, result.var81);
  ptr = parse(ptr, result.var82);
  ptr = parse(ptr, result.var83);
  ptr = parse(ptr, result.var84);
  ptr = parse(ptr, result.var85);
  ptr = parse(ptr, result.var86);
  ptr = parse(ptr, result.var87);
  ptr = parse(ptr, result.var88);
  ptr = parse(ptr, result.var89);
  ptr = parse(ptr, result.var90);
  ptr = parse(ptr, result.var91);
  ptr = parse(ptr, result.var92);
  ptr = parse(ptr, result.var93);
  ptr = parse(ptr, result.var94);
  ptr = parse(ptr, result.var95);
  ptr = parse(ptr, result.var96);
  ptr = parse(ptr, result.var97);
  ptr = parse(ptr, result.var98);
  ptr = parse(ptr, result.var99);
  ptr = parse(ptr, result.var100);
  ptr = parse(ptr, result.var101);
  ptr = parse(ptr, result.var102);
  ptr = parse(ptr, result.var103);
  ptr = parse(ptr, result.var104);
  ptr = parse(ptr, result.var105);
  ptr = parse(ptr, result.var106);
  ptr = parse(ptr, result.var107);
  ptr = parse(ptr, result.var108);
  ptr = parse(ptr, result.var109);
  ptr = parse(ptr, result.var110);
  ptr = parse(ptr, result.var111);
  ptr = parse(ptr, result.var112);
  ptr = parse(ptr, result.var113);
  ptr = parse(ptr, result.var114);
  ptr = parse(ptr, result.var115);
  ptr = parse(ptr, result.var116);
  ptr = parse(ptr, result.var117);
  ptr = parse(ptr, result.var118);
  ptr = parse(ptr, result.var119);
  ptr = parse(ptr, result.var120);
  ptr = parse(ptr, result.var121);
  ptr = parse(ptr, result.var122);
  ptr = parse(ptr, result.var123);
  ptr = parse(ptr, result.var124);
  ptr = parse(ptr, result.var125);
  ptr = parse(ptr, result.var126);
  ptr = parse(ptr, result.var127);
  ptr = parse(ptr, result.var128);
  ptr = parse(ptr, result.var129);
  ptr = parse(ptr, result.var130);
  ptr = parse(ptr, result.var131);
  ptr = parse(ptr, result.var132);
  ptr = parse(ptr, result.var133);
  ptr = parse(ptr, result.var134);
  ptr = parse(ptr, result.var135);
  ptr = parse(ptr, result.var136);
  ptr = parse(ptr, result.var137);
  ptr = parse(ptr, result.var138);
  ptr = parse(ptr, result.var139);
  ptr = parse(ptr, result.var140);
  ptr = parse(ptr, result.var141);
  ptr = parse(ptr, result.var142);
  ptr = parse(ptr, result.var143);
  ptr = parse(ptr, result.var144);
  ptr = parse(ptr, result.var145);
  ptr = parse(ptr, result.var146);
  ptr = parse(ptr, result.var147);
  ptr = parse(ptr, result.var148);
  ptr = parse(ptr, result.var149);
  ptr = parse(ptr, result.var150);
  ptr = parse(ptr, result.var151);
  ptr = parse(ptr, result.var152);
  ptr = parse(ptr, result.var153);
  ptr = parse(ptr, result.var154);
  ptr = parse(ptr, result.var155);
  ptr = parse(ptr, result.var156);
  ptr = parse(ptr, result.var157);
  ptr = parse(ptr, result.var158);
  ptr = parse(ptr, result.var159);
  ptr = parse(ptr, result.var160);
  return result;
}

void print(const ParsedLine &info) {
  cout << info.var0 << '\t';
  cout << info.var1 << '\t';
  cout << info.var2 << '\t';
  cout << info.var3 << '\t';
  cout << info.var4 << '\t';
  cout << info.var5 << '\t';
  cout << info.var6 << '\t';
  cout << info.var7 << '\t';
  cout << info.var8 << '\t';
  cout << info.var9 << '\t';
  cout << info.var10 << '\t';
  cout << info.var11 << '\t';
  cout << info.var12 << '\t';
  cout << info.var13 << '\t';
  cout << info.var14 << '\t';
  cout << info.var15 << '\t';
  cout << info.var16 << '\t';
  cout << info.var17 << '\t';
  cout << info.var18 << '\t';
  cout << info.var19 << '\t';
  cout << info.var20 << '\t';
  cout << info.var21 << '\t';
  cout << info.var22 << '\t';
  cout << info.var23 << '\t';
  cout << info.var24 << '\t';
  cout << info.var25 << '\t';
  cout << info.var26 << '\t';
  cout << info.var27 << '\t';
  cout << info.var28 << '\t';
  cout << info.var29 << '\t';
  cout << info.var30 << '\t';
  cout << info.var31 << '\t';
  cout << info.var32 << '\t';
  cout << info.var33 << '\t';
  cout << info.var34 << '\t';
  cout << info.var35 << '\t';
  cout << info.var36 << '\t';
  cout << info.var37 << '\t';
  cout << info.var38 << '\t';
  cout << info.var39 << '\t';
  cout << info.var40 << '\t';
  cout << info.var41 << '\t';
  cout << info.var42 << '\t';
  cout << info.var43 << '\t';
  cout << info.var44 << '\t';
  cout << info.var45 << '\t';
  cout << info.var46 << '\t';
  cout << info.var47 << '\t';
  cout << info.var48 << '\t';
  cout << info.var49 << '\t';
  cout << info.var50 << '\t';
  cout << info.var51 << '\t';
  cout << info.var52 << '\t';
  cout << info.var53 << '\t';
  cout << info.var54 << '\t';
  cout << info.var55 << '\t';
  cout << info.var56 << '\t';
  cout << info.var57 << '\t';
  cout << info.var58 << '\t';
  cout << info.var59 << '\t';
  cout << info.var60 << '\t';
  cout << info.var61 << '\t';
  cout << info.var62 << '\t';
  cout << info.var63 << '\t';
  cout << info.var64 << '\t';
  cout << info.var65 << '\t';
  cout << info.var66 << '\t';
  cout << info.var67 << '\t';
  cout << info.var68 << '\t';
  cout << info.var69 << '\t';
  cout << info.var70 << '\t';
  cout << info.var71 << '\t';
  cout << info.var72 << '\t';
  cout << info.var73 << '\t';
  cout << info.var74 << '\t';
  cout << info.var75 << '\t';
  cout << info.var76 << '\t';
  cout << info.var77 << '\t';
  cout << info.var78 << '\t';
  cout << info.var79 << '\t';
  cout << info.var80 << '\t';
  cout << info.var81 << '\t';
  cout << info.var82 << '\t';
  cout << info.var83 << '\t';
  cout << info.var84 << '\t';
  cout << info.var85 << '\t';
  cout << info.var86 << '\t';
  cout << info.var87 << '\t';
  cout << info.var88 << '\t';
  cout << info.var89 << '\t';
  cout << info.var90 << '\t';
  cout << info.var91 << '\t';
  cout << info.var92 << '\t';
  cout << info.var93 << '\t';
  cout << info.var94 << '\t';
  cout << info.var95 << '\t';
  cout << info.var96 << '\t';
  cout << info.var97 << '\t';
  cout << info.var98 << '\t';
  cout << info.var99 << '\t';
  cout << info.var100 << '\t';
  cout << info.var101 << '\t';
  cout << info.var102 << '\t';
  cout << info.var103 << '\t';
  cout << info.var104 << '\t';
  cout << info.var105 << '\t';
  cout << info.var106 << '\t';
  cout << info.var107 << '\t';
  cout << info.var108 << '\t';
  cout << info.var109 << '\t';
  cout << info.var110 << '\t';
  cout << info.var111 << '\t';
  cout << info.var112 << '\t';
  cout << info.var113 << '\t';
  cout << info.var114 << '\t';
  cout << info.var115 << '\t';
  cout << info.var116 << '\t';
  cout << info.var117 << '\t';
  cout << info.var118 << '\t';
  cout << info.var119 << '\t';
  cout << info.var120 << '\t';
  cout << info.var121 << '\t';
  cout << info.var122 << '\t';
  cout << info.var123 << '\t';
  cout << info.var124 << '\t';
  cout << info.var125 << '\t';
  cout << info.var126 << '\t';
  cout << info.var127 << '\t';
  cout << info.var128 << '\t';
  cout << info.var129 << '\t';
  cout << info.var130 << '\t';
  cout << info.var131 << '\t';
  cout << info.var132 << '\t';
  cout << info.var133 << '\t';
  cout << info.var134 << '\t';
  cout << info.var135 << '\t';
  cout << info.var136 << '\t';
  cout << info.var137 << '\t';
  cout << info.var138 << '\t';
  cout << info.var139 << '\t';
  cout << info.var140 << '\t';
  cout << info.var141 << '\t';
  cout << info.var142 << '\t';
  cout << info.var143 << '\t';
  cout << info.var144 << '\t';
  cout << info.var145 << '\t';
  cout << info.var146 << '\t';
  cout << info.var147 << '\t';
  cout << info.var148 << '\t';
  cout << info.var149 << '\t';
  cout << info.var150 << '\t';
  cout << info.var151 << '\t';
  cout << info.var152 << '\t';
  cout << info.var153 << '\t';
  cout << info.var154 << '\t';
  cout << info.var155 << '\t';
  cout << info.var156 << '\t';
  cout << info.var157 << '\t';
  cout << info.var158 << '\t';
  cout << info.var159 << '\t';
  cout << info.var160 << '\n';
}