#!/usr/bin/env python3
import sys

def strToMacroCase(str):
  result = ""
  for c in str:
    if c.isupper():
      result += '_'
    result += c
  return result.upper()

if len(sys.argv) < 2:
  print("Usage: {} <PascalCaseClassName>".format(sys.argv[0]))
  exit(0)

pascalCaseClassName = sys.argv[1]
camelCaseClassName = pascalCaseClassName[0].lower() + pascalCaseClassName[1:]
macroCaseClassName = strToMacroCase(camelCaseClassName)

cmakeFilePath = '../Combined/CMakeLists.txt'
projectFileSourceLine = '    "src/packet/parsing/{}.cpp"'
projectFileHeaderLine = '    "src/packet/parsing/{}.hpp"'

# Append the file paths to the project file
with open(cmakeFilePath, 'a') as projectFile:
  projectFile.write('\n'+projectFileSourceLine.format(camelCaseClassName))
  projectFile.write('\n'+projectFileHeaderLine.format(camelCaseClassName))
  print("Lines written to {}, please go move them into the proper location".format(cmakeFilePath))

packetParserFilePath = '../Combined/src/packet/parsing/packetParser.cpp'
packetParserHandlerData = '#include "{camelClassName}.hpp"\n'\
                          '      case Opcode::k{pascalClassName}:\n'\
                          '        return std::make_unique<{pascalClassName}>(packet);'
# Append the switch case for the packet parser
with open(packetParserFilePath, 'a') as packetParserFile:
  packetParserFile.write('\n'+packetParserHandlerData.format(camelClassName=camelCaseClassName, pascalClassName=pascalCaseClassName))
  print("Lines written to {}, please go move them into the proper location".format(packetParserFilePath))

sourceFilePath = '../Combined/src/packet/parsing/{}.cpp'
headerFilePath = '../Combined/src/packet/parsing/{}.hpp'
headerTemplate = '#ifndef PACKET_PARSING_{macroClassName}_HPP_\n'\
                 '#define PACKET_PARSING_{macroClassName}_HPP_\n'\
                 '\n'\
                 '#include "packet/parsing/parsedPacket.hpp"\n'\
                 '\n'\
                 'namespace packet::parsing {{\n'\
                 '\n'\
                 'class {pascalClassName} : public ParsedPacket {{\n'\
                 'public:\n'\
                 '  {pascalClassName}(const PacketContainer &packet);\n'\
                 'private:\n'\
                 '}};\n'\
                 '\n'\
                 '}} // namespace packet::parsing\n'\
                 '\n'\
                 '#endif // PACKET_PARSING_{macroClassName}_HPP_'
sourceTemplate = '#include "{camelClassName}.hpp"\n'\
                 '\n'\
                 'namespace packet::parsing {{\n'\
                 '\n'\
                 '{pascalClassName}::{pascalClassName}(const PacketContainer &packet) :\n'\
                 '      ParsedPacket(packet) {{\n'\
                 '  StreamUtility stream = packet.data;\n'\
                 '}}\n'\
                 '\n'\
                 '}} // namespace packet::parsing'

with open(sourceFilePath.format(camelCaseClassName), 'w') as sourceFile:
  sourceFile.write(sourceTemplate.format(camelClassName=camelCaseClassName, pascalClassName=pascalCaseClassName))
  print("Source file ({}) created, please double check".format(sourceFilePath.format(camelCaseClassName)))

with open(headerFilePath.format(camelCaseClassName), 'w') as headerFile:
  headerFile.write(headerTemplate.format(macroClassName=macroCaseClassName, pascalClassName=pascalCaseClassName))
  print("Header file ({}) created, please double check".format(headerFilePath.format(camelCaseClassName)))
