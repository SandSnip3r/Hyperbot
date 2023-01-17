#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

std::array<std::string, 3> splitLine(const string &line, const char delim) {
  std::array<std::string, 3> result;
  const auto posOfFirstDelim = line.find(delim);
  if (posOfFirstDelim == std::string::npos) {
    throw std::runtime_error("No first delim in line");
  }
  const auto posOfSecondDelim = line.find(delim, posOfFirstDelim+1);
  if (posOfSecondDelim == std::string::npos) {
    throw std::runtime_error("No second delim in line");
  }
  const int numOfCharsInFirst = posOfFirstDelim;
  const int numOfCharsInSecond = posOfSecondDelim - posOfFirstDelim - 1;
  return {line.substr(0, numOfCharsInFirst), line.substr(posOfFirstDelim+1, numOfCharsInSecond), line.substr(posOfSecondDelim+1)};
}

enum class MagOpt {
  kAstral, // MATTR_ASTRAL
  kImmortal, // MATTR_ATHANASIA
  kDurability, // MATTR_DUR
  kParryRate, // MATTR_ER
  kBlockParry, // MATTR_EVADE_BLOCK
  kCriticalParry, // MATTR_EVADE_CRITICAL
  kHp, // MATTR_HP
  kAttackRate, // MATTR_HR
  kInt, // MATTR_INT
  kLuck, // MATTR_LUCK
  kMp, // MATTR_MP
  kResistBurn, // MATTR_RESIST_BURN
  kResistShock, // MATTR_RESIST_ESHOCK
  kResistIce, // MATTR_RESIST_FROSTBITE
  kResistPoison, // MATTR_RESIST_POISON
  kResistZombie, // MATTR_RESIST_ZOMBIE
  kSteady, // MATTR_SOLID
  kStr, // MATTR_STR
};

MagOpt magOptFromString(const std::string &magOptName) {
  if (magOptName == "MATTR_ASTRAL") {
    return MagOpt::kAstral;
  }
  if (magOptName == "MATTR_ATHANASIA") {
    return MagOpt::kImmortal;
  }
  if (magOptName == "MATTR_DUR") {
    return MagOpt::kDurability;
  }
  if (magOptName == "MATTR_ER") {
    return MagOpt::kParryRate;
  }
  if (magOptName == "MATTR_EVADE_BLOCK") {
    return MagOpt::kBlockParry;
  }
  if (magOptName == "MATTR_EVADE_CRITICAL") {
    return MagOpt::kCriticalParry;
  }
  if (magOptName == "MATTR_HP") {
    return MagOpt::kHp;
  }
  if (magOptName == "MATTR_HR") {
    return MagOpt::kAttackRate;
  }
  if (magOptName == "MATTR_INT") {
    return MagOpt::kInt;
  }
  if (magOptName == "MATTR_LUCK") {
    return MagOpt::kLuck;
  }
  if (magOptName == "MATTR_MP") {
    return MagOpt::kMp;
  }
  if (magOptName == "MATTR_RESIST_BURN") {
    return MagOpt::kResistBurn;
  }
  if (magOptName == "MATTR_RESIST_ESHOCK") {
    return MagOpt::kResistShock;
  }
  if (magOptName == "MATTR_RESIST_FROSTBITE") {
    return MagOpt::kResistIce;
  }
  if (magOptName == "MATTR_RESIST_POISON") {
    return MagOpt::kResistPoison;
  }
  if (magOptName == "MATTR_RESIST_ZOMBIE") {
    return MagOpt::kResistZombie;
  }
  if (magOptName == "MATTR_SOLID") {
    return MagOpt::kSteady;
  }
  if (magOptName == "MATTR_STR") {
    return MagOpt::kStr;
  }
  throw std::runtime_error("Unknown mag opt");
}

class MagOptIdMachine {
public:
  static const MagOptIdMachine& instance() {
    static MagOptIdMachine m;
    return m;
  }

  uint64_t getMagParamId(MagOpt magOpt, int degree) const {
    degree = std::clamp(degree, 1, 12);
    return magOptToDegreeToIdMap_.at(magOpt).at(degree);
  }
private:
  MagOptIdMachine() {
    ifstream magicOptionDataFile("trimmed_magicoption.txt");
    if (!magicOptionDataFile) {
      throw std::runtime_error("Cannot open magic option data file");
    }
    string line;
    while (getline(magicOptionDataFile, line)) {
      auto linePieces = splitLine(line, ' ');
      try {
        int id = stoi(linePieces.at(0));
        const std::string &name = linePieces.at(1);
        auto magOpt = magOptFromString(name);
        int degree = stoi(linePieces.at(2));
        magOptToDegreeToIdMap_[magOpt][degree] = id;
      } catch (std::exception &ex) {
        // Ignoring failures
        // cout << "Exception: " << ex.what() << endl;;
      }
    }
  }

  std::map<MagOpt, std::map<int, int>> magOptToDegreeToIdMap_;
};

uint64_t percentToVal(double percent) {
  // Convert [0,100] to [0,31]
  percent = std::clamp(percent/100, 0.0, 1.0);
  percent *= 31;
  return std::round(percent);
}

class Item {
public:
  Item(int degree);
  uint64_t getVariance() const;
  const std::vector<uint64_t>& getMagicOptions() const;
  string getUpdateQuery() const;
  void addBlue(MagOpt opt, uint64_t value);
  virtual ~Item() {}
protected:
  void addWhite(uint8_t index, double targetPercent);
private:
  const int degree_;
  uint64_t variance_{0};
  std::vector<uint64_t> magicOptions_;
};

Item::Item(int degree) : degree_(degree) {}

uint64_t Item::getVariance() const {
  return variance_;
}

const std::vector<uint64_t>& Item::getMagicOptions() const {
  return magicOptions_;
}

void printMagicOptionsAsQuery(const std::vector<uint64_t> &magicOptions) {
  std::cout << std::endl;
}

string Item::getUpdateQuery() const {
  stringstream ss;
  ss << "Variance=" << to_string(variance_);
  ss << ", MagParamNum=" << magicOptions_.size();
  for (int i=0; i<magicOptions_.size(); ++i) {
    ss << ", MagParam" << i+1 << "=" << magicOptions_.at(i);
  }
  return ss.str();
}

void Item::addBlue(MagOpt opt, uint64_t value) {
  const auto magParamId = MagOptIdMachine::instance().getMagParamId(opt, degree_);
  magicOptions_.push_back(value<<32 | magParamId&0xFFFFFFFF);
}

void Item::addWhite(uint8_t index, double targetPercent) {
  constexpr uint8_t kWidthPerField{5};
  variance_ |= (percentToVal(targetPercent) & ((1<<kWidthPerField) - 1)) << (kWidthPerField*index);
}

// If a weapon is physical or magical only, adding the opposite white states has no effect (that I can see)
class Weapon : public Item {
  using Item::Item;
public:
  enum class Stat : uint8_t {
    kDurability = 0,
    kPhysicalReinforce = 1,
    kMagicalReinforce = 2,
    kAttackRate = 3,
    kPhysicalDamage = 4,
    kMagicalDamage = 5,
    kCritical = 6
  };
  void addWhite(Stat stat, double percent) {
    Item::addWhite(static_cast<uint8_t>(stat), percent);
  }
  static Weapon createFullBlue(double whitePercent, int degree) {
    Weapon weapon(degree);
    weapon.addWhite(Weapon::Stat::kDurability, whitePercent);
    weapon.addWhite(Weapon::Stat::kPhysicalReinforce, whitePercent);
    weapon.addWhite(Weapon::Stat::kMagicalReinforce, whitePercent);
    weapon.addWhite(Weapon::Stat::kAttackRate, whitePercent);
    weapon.addWhite(Weapon::Stat::kPhysicalDamage, whitePercent);
    weapon.addWhite(Weapon::Stat::kMagicalDamage, whitePercent);
    weapon.addWhite(Weapon::Stat::kCritical, whitePercent);
    weapon.addBlue(MagOpt::kAttackRate, 60);
    weapon.addBlue(MagOpt::kBlockParry, 100);
    weapon.addBlue(MagOpt::kDurability, 200);
    weapon.addBlue(MagOpt::kInt, 7);
    weapon.addBlue(MagOpt::kStr, 7);
    return weapon;
  }
};

class Armor : public Item {
  using Item::Item;
public:
  enum class Stat : uint8_t {
    kDurability = 0,
    kPhysicalReinforce = 1,
    kMagicalReinforce = 2,
    kPhysicalDamage = 3,
    kMagicalDamage = 4,
    kParryRate = 5
  };
  void addWhite(Stat stat, double percent) {
    Item::addWhite(static_cast<uint8_t>(stat), percent);
  }
  static Armor createFullBlue(double whitePercent, int degree, bool isCore) {
    Armor armor(degree);
    armor.addWhite(Armor::Stat::kDurability, whitePercent);
    armor.addWhite(Armor::Stat::kPhysicalReinforce, whitePercent);
    armor.addWhite(Armor::Stat::kMagicalReinforce, whitePercent);
    armor.addWhite(Armor::Stat::kPhysicalDamage, whitePercent);
    armor.addWhite(Armor::Stat::kMagicalDamage, whitePercent);
    armor.addWhite(Armor::Stat::kParryRate, whitePercent);
    armor.addBlue(MagOpt::kParryRate, 60);
    armor.addBlue(MagOpt::kDurability, 200);
    armor.addBlue(MagOpt::kInt, 7);
    armor.addBlue(MagOpt::kStr, 7);
    if (isCore) {
      armor.addBlue(MagOpt::kHp, 1300);
      armor.addBlue(MagOpt::kMp, 1300);
    }
    return armor;
  }
};

class Shield : public Item {
  using Item::Item;
public:
  enum class Stat : uint8_t {
    kDurability = 0,
    kPhysicalReinforce = 1,
    kMagicalReinforce = 2,
    kBlockRate = 3,
    kPhysicalDefense = 4,
    kMagicalDefense = 5
  };
  void addWhite(Stat stat, double percent) {
    Item::addWhite(static_cast<uint8_t>(stat), percent);
  }
  static Shield createFullBlue(double whitePercent, int degree) {
    Shield shield(degree);
    shield.addWhite(Shield::Stat::kDurability, whitePercent);
    shield.addWhite(Shield::Stat::kPhysicalReinforce, whitePercent);
    shield.addWhite(Shield::Stat::kMagicalReinforce, whitePercent);
    shield.addWhite(Shield::Stat::kBlockRate, whitePercent);
    shield.addWhite(Shield::Stat::kPhysicalDefense, whitePercent);
    shield.addWhite(Shield::Stat::kMagicalDefense, whitePercent);
    shield.addBlue(MagOpt::kCriticalParry, 100);
    shield.addBlue(MagOpt::kInt, 7);
    shield.addBlue(MagOpt::kStr, 7);
    shield.addBlue(MagOpt::kDurability, 200);
    return shield;
  }
};

class Accessory : public Item {
  using Item::Item;
public:
  enum class Stat : uint8_t {
    kPhysicalAbsorption = 0,
    kMagicalAbsorption = 1
  };
  void addWhite(Stat stat, double percent) {
    Item::addWhite(static_cast<uint8_t>(stat), percent);
  }
  static Accessory createFullBlue(double whitePercent, int degree) {
    Accessory accessory(degree);
    accessory.addWhite(Accessory::Stat::kPhysicalAbsorption, whitePercent);
    accessory.addWhite(Accessory::Stat::kMagicalAbsorption, whitePercent);
    accessory.addBlue(MagOpt::kResistBurn, 20);
    accessory.addBlue(MagOpt::kResistShock, 20);
    accessory.addBlue(MagOpt::kResistIce, 20);
    accessory.addBlue(MagOpt::kResistPoison, 20);
    accessory.addBlue(MagOpt::kResistZombie, 20);
    accessory.addBlue(MagOpt::kInt, 7);
    accessory.addBlue(MagOpt::kStr, 7);
    return accessory;
  }
};

class SetBuilder {
public:
  SetBuilder(int charId, int degree) : charId_(charId), degree_(degree) {}
  void setTargetStatPercent(double percent) { statPercent_ = percent; }
  void setWeaponSlots(const std::vector<int> &slots) { weaponSlots_ = slots; }
  void setArmorCoreSlots(const std::vector<int> &slots) { armorCoreSlots_ = slots; }
  void setArmorLimbSlots(const std::vector<int> &slots) { armorLimbSlots_ = slots; }
  void setShieldSlots(const std::vector<int> &slots) { shieldSlots_ = slots; }
  void setAccessorySlots(const std::vector<int> &slots) { accessorySlots_ = slots; }
  void build() const;

  string getTypeQuery(const Item *item, const vector<int> &slots) const {
    stringstream slotsSS;
    if (!slots.empty()) {
      slotsSS << slots.at(0);
    }
    for (int i=1; i<slots.size(); ++i) {
      slotsSS << ',' << slots.at(i);
    }
    string queryPiece1 = 
R"(UPDATE [SRO_VT_SHARD].[dbo].[_Items]
  SET )";
    string queryPiece2 = R"(
  WHERE ID64 IN (
  SELECT ItemID
    FROM [SRO_VT_SHARD].[dbo].[_Inventory]
    WHERE CharID=)" + to_string(charId_) + R"( AND Slot IN ()" + slotsSS.str() + R"()
))";
    stringstream result;
    result << queryPiece1 << item->getUpdateQuery() << queryPiece2 << endl;
    return result.str();
  }
private:
  const int charId_;
  const int degree_;
  double statPercent_{0.0};
  std::vector<int> weaponSlots_;
  std::vector<int> armorCoreSlots_;
  std::vector<int> armorLimbSlots_;
  std::vector<int> shieldSlots_;
  std::vector<int> accessorySlots_;
};

void SetBuilder::build() const {
  // Assuming full blue
  if (!weaponSlots_.empty()) {
    cout << "-- weapon:" << endl;
    auto weapon = Weapon::createFullBlue(statPercent_, degree_);
    cout << getTypeQuery(&weapon, weaponSlots_) << endl;
  }
  if (!armorCoreSlots_.empty()) {
    cout << "-- armor core:" << endl;
    auto armor = Armor::createFullBlue(statPercent_, degree_,true);
    cout << getTypeQuery(&armor, armorCoreSlots_) << endl;
  }
  if (!armorLimbSlots_.empty()) {
    cout << "-- armor limb:" << endl;
    auto armor = Armor::createFullBlue(statPercent_, degree_,false);
    cout << getTypeQuery(&armor, armorLimbSlots_) << endl;
  }
  if (!shieldSlots_.empty()) {
    cout << "-- shield:" << endl;
    auto shield = Shield::createFullBlue(statPercent_, degree_);
    cout << getTypeQuery(&shield, shieldSlots_) << endl;
  }
  if (!accessorySlots_.empty()) {
    cout << "-- accessory:" << endl;
    auto accessory = Accessory::createFullBlue(statPercent_, degree_);
    cout << getTypeQuery(&accessory, accessorySlots_) << endl;
  }
}

int main() {
  const double targetPercent = 100;
  const int kCharId = 6725;
  const int kDegree{11};
  SetBuilder setBuilder(kCharId, kDegree);
  setBuilder.setTargetStatPercent(targetPercent);
  setBuilder.setWeaponSlots({17});
  setBuilder.setArmorCoreSlots({0,1,4});
  setBuilder.setArmorLimbSlots({2,3,5});
  setBuilder.setShieldSlots({22});
  setBuilder.setAccessorySlots({9,10,11,12});
  setBuilder.build();
  return 0;
}