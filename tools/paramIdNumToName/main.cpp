#include <iostream>
#include <memory>

using namespace std;

void printNum(uint32_t num) {
  char c[5];
  c[3] = num & 0xFF;
  num >>= 8;
  c[2] = num & 0xFF;
  num >>= 8;
  c[1] = num & 0xFF;
  num >>= 8;
  c[0] = num & 0xFF;
  int zeroCount=0;
  while (c[zeroCount] == 0) {
    ++zeroCount;
  }
  for (int i=zeroCount; i<4; ++i) {
    c[i-zeroCount] = c[i];
  }
  c[4-zeroCount] = 0;
  cout << c << '\n';
}

int main() {

  printNum(1734702198);
  printNum(28003);
  printNum(6386804);
  printNum(1952803941);
  
  
  
  return 0;
}