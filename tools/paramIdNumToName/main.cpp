#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace std;

/*
A string like "reqi" is packed into a uint32_t like:
r e q i
Which ends up being:
1919250793 (base 10)
Which is the following in hex:
0x72657169
Ascii values:
r: 0x72
e: 0x65
q: 0x71
i: 0x69
*/

std::string toString(const char *numStr) {
  uint32_t num = atoll(numStr);
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
  return c;
}

uint32_t toNum(const char *str) {
  int len = strlen(str);
  uint32_t num = 0;
  for (int i=0; i<len; ++i) {
    num <<= 8;
    num |= static_cast<uint8_t>(str[i]);
  }
  return num;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "This tool can convert param names to numbers or vice versa" << std::endl;
    std::cout << "Usage: " << argv[0] << " (param_name|param_number)" << std::endl;
    return 1;
  }
  if (std::isdigit(argv[1][0])) {
    std::cout << toString(argv[1]) << std::endl;
  } else if (std::isalpha(argv[1][0])) {
    std::cout << toNum(argv[1]) << std::endl;
  } else {
    std::cout << "Argument is not a name nor a number \"" << argv[1] << "\"" << std::endl;
  }
  return 0;
}