#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace std;

struct Item {
public:
  Item(const std::string &n) : name(n) {}
  std::string name;
};

class Storage {
public:
  Storage() = default;
  Storage(uint8_t size) : items_(static_cast<size_t>(size)) {}

  //=====================================
  //============Non-modifying============
  //=====================================

  bool hasItem(uint8_t slot) const {
    boundsCheck(slot, "hasItem");
    return (items_[slot] != nullptr);
  }

  Item* getItem(uint8_t slot) {
    boundsCheck(slot, "getItem");
    return items_[slot].get();
  }

  const Item* getItem(uint8_t slot) const {
    boundsCheck(slot, "getItem");
    return items_[slot].get();
  }

  uint8_t size() const { return static_cast<uint8_t>(items_.size()); }

  //=====================================
  //==============Modifying==============
  //=====================================

  void addItem(uint8_t slot, std::shared_ptr<Item> item) {
    boundsCheck(slot, "addItem");
    if (hasItem(slot)) {
      throw std::runtime_error("Storage::addItem: Item already exists in slot "+std::to_string(slot));
    }
    items_[slot] = item;
  }

  void resize(uint8_t newSize) {
    if (newSize < items_.size()) {
      throw std::runtime_error("Storage::resize: Trying to reduce size "+std::to_string(items_.size())+" -> "+std::to_string(newSize));
    } else if (newSize > items_.size()) {
      items_.resize(newSize);
    }
  }
  
  //=====================================
private:
  std::vector<std::shared_ptr<Item>> items_;

  void boundsCheck(uint8_t slot, const std::string &funcName) const {
    if (slot >= items_.size()) {
      throw std::runtime_error("Storage::"+funcName+": Out of bounds "+std::to_string(slot)+" >= "+std::to_string(items_.size()));
    }
  }
};

int main() {
  Storage s;
  s.resize(20);
  s.addItem(0, make_shared<Item>("First"));
  s.addItem(3, make_shared<Item>("Second"));
  s.resize(40);
  s.addItem(30, make_shared<Item>("Third"));
  for (int i=0; i<s.size(); ++i) {
    if (s.hasItem(i)) {
      const auto *item = s.getItem(i);
      cout << i << ' ' << item->name << '\n';
    }
  }
  return 0;
}