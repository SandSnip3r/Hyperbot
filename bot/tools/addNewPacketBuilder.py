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

cmakeFilePath = '../Hyperbot/CMakeLists.txt'
projectFileSourceLine = '    "src/packet/building/{}.cpp"'
projectFileHeaderLine = '    "src/packet/building/{}.hpp"'

# Append the file paths to the project file
with open(cmakeFilePath, 'a') as projectFile:
  projectFile.write('\n'+projectFileSourceLine.format(camelCaseClassName))
  projectFile.write('\n'+projectFileHeaderLine.format(camelCaseClassName))
  print("Lines written to {}, please go move them into the proper location".format(cmakeFilePath))

sourceFilePath = '../Hyperbot/src/packet/building/{}.cpp'
headerFilePath = '../Hyperbot/src/packet/building/{}.hpp'
headerTemplate = '#ifndef PACKET_BUILDING_{macroClassName}_HPP_\n'\
                 '#define PACKET_BUILDING_{macroClassName}_HPP_\n'\
                 '\n'\
                 '#include "packet/opcode.hpp"\n'\
                 '\n'\
                 '#include "../../shared/silkroad_security.h"\n'\
                 '\n'\
                 'namespace packet::building {{\n'\
                 '\n'\
                 'class {pascalClassName} {{\n'\
                 'private:\n'\
                 '  static const Opcode kOpcode_ = Opcode::k{pascalClassName};\n'\
                 '  static const bool kEncrypted_ = false;\n'\
                 '  static const bool kMassive_ = false;\n'\
                 'public:\n'\
                 '  static PacketContainer packet();\n'\
                 '}};\n'\
                 '\n'\
                 '}} // namespace packet::building\n'\
                 '\n'\
                 '#endif // PACKET_BUILDING_{macroClassName}_HPP_'
sourceTemplate = '#include "{camelClassName}.hpp"\n'\
                 '\n'\
                 'namespace packet::building {{\n'\
                 '\n'\
                 'PacketContainer {pascalClassName}::packet() {{\n'\
                 '  StreamUtility stream;\n'\
                 '  stream.Write<>();\n'\
                 '  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));\n'\
                 '}}\n'\
                 '\n'\
                 '}} // namespace packet::building'

with open(sourceFilePath.format(camelCaseClassName), 'w') as sourceFile:
  sourceFile.write(sourceTemplate.format(camelClassName=camelCaseClassName, pascalClassName=pascalCaseClassName))
  print("Source file ({}) created, please double check".format(sourceFilePath.format(camelCaseClassName)))

with open(headerFilePath.format(camelCaseClassName), 'w') as headerFile:
  headerFile.write(headerTemplate.format(macroClassName=macroCaseClassName, pascalClassName=pascalCaseClassName))
  print("Header file ({}) created, please double check".format(headerFilePath.format(camelCaseClassName)))
