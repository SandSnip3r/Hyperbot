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

cmakeFilePath = '../bot/CMakeLists.txt'
projectFileSourceLine = '  src/state/machine/{}.cpp'
projectFileHeaderLine = '  src/state/machine/{}.hpp'

# Append the file paths to the project file
with open(cmakeFilePath, 'a') as projectFile:
  projectFile.write('\n'+projectFileSourceLine.format(camelCaseClassName))
  projectFile.write('\n'+projectFileHeaderLine.format(camelCaseClassName))
  print("Lines written to {}, please go move them into the proper location".format(cmakeFilePath))

sourceFilePath = '../bot/src/state/machine/{}.cpp'
headerFilePath = '../bot/src/state/machine/{}.hpp'
headerTemplate = '#ifndef STATE_MACHINE_{macroClassName}_HPP_\n'\
                 '#define STATE_MACHINE_{macroClassName}_HPP_\n'\
                 '\n'\
                 '#include "event/event.hpp"\n'\
                 '#include "state/machine/stateMachine.hpp"\n'\
                 '\n'\
                 '#include <string>\n'\
                 '\n'\
                 'namespace state::machine {{\n'\
                 '\n'\
                 'class {pascalClassName} : public StateMachine {{\n'\
                 'public:\n'\
                 '  {pascalClassName}(StateMachine *parent);\n'\
                 '  ~{pascalClassName}() override;\n'\
                 '  Status onUpdate(const event::Event *event) override;\n'\
                 'private:\n'\
                 '  static inline std::string kName{{"{pascalClassName}"}};\n'\
                 '}};\n'\
                 '\n'\
                 '}} // namespace state::machine\n'\
                 '\n'\
                 '#endif // STATE_MACHINE_{macroClassName}_HPP_\n'

sourceTemplate = '#include "{camelClassName}.hpp"\n'\
                 '\n'\
                 '#include "bot.hpp"\n'\
                 '\n'\
                 '#include <absl/log/log.h>\n'\
                 '\n'\
                 'namespace state::machine {{\n'\
                 '\n'\
                 '{pascalClassName}::{pascalClassName}(StateMachine *parent) : StateMachine(parent) {{\n'\
                 '}}\n'\
                 '\n'\
                 '{pascalClassName}::~{pascalClassName}() {{\n'\
                 '}}\n'\
                 '\n'\
                 'Status {pascalClassName}::onUpdate(const event::Event *event) {{\n'\
                 '  return Status::kNotDone;\n'\
                 '}}\n'\
                 '\n'\
                 '}} // namespace state::machine\n'

with open(sourceFilePath.format(camelCaseClassName), 'w') as sourceFile:
  sourceFile.write(sourceTemplate.format(camelClassName=camelCaseClassName, pascalClassName=pascalCaseClassName))
  print("Source file ({}) created, please double check".format(sourceFilePath.format(camelCaseClassName)))

with open(headerFilePath.format(camelCaseClassName), 'w') as headerFile:
  headerFile.write(headerTemplate.format(macroClassName=macroCaseClassName, pascalClassName=pascalCaseClassName))
  print("Header file ({}) created, please double check".format(headerFilePath.format(camelCaseClassName)))
