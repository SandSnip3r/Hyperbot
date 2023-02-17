#include "builtInTypeWidget.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>

namespace pb = google::protobuf;

namespace {

int parseEnumFromWidget(const QWidget *widget);
bool parseBoolFromWidget(const QWidget *widget);

template<typename T>
T parseDataFromWidget(const QWidget *widget);

template<>
std::string parseDataFromWidget<std::string>(const QWidget *widget) {
  const QLineEdit *lineEdit = dynamic_cast<const QLineEdit*>(widget);
  if (lineEdit == nullptr) {
    throw std::runtime_error("Parsing data from widget, std::string, but widget is not a lineedit");
  }
  return lineEdit->text().toStdString();
}

template<>
float parseDataFromWidget<float>(const QWidget *widget) {
  const QLineEdit *lineEdit = dynamic_cast<const QLineEdit*>(widget);
  if (lineEdit == nullptr) {
    throw std::runtime_error("Parsing data from widget, float, but widget is not a lineedit");
  }
  return lineEdit->text().toFloat();
}

template<>
double parseDataFromWidget<double>(const QWidget *widget) {
  const QLineEdit *lineEdit = dynamic_cast<const QLineEdit*>(widget);
  if (lineEdit == nullptr) {
    throw std::runtime_error("Parsing data from widget, double, but widget is not a lineedit");
  }
  return lineEdit->text().toDouble();
}

template<>
int32_t parseDataFromWidget<int32_t>(const QWidget *widget) {
  const QLineEdit *lineEdit = dynamic_cast<const QLineEdit*>(widget);
  if (lineEdit == nullptr) {
    throw std::runtime_error("Parsing data from widget, int32_t, but widget is not a lineedit");
  }
  if (std::is_same_v<int, int32_t>) {
    return lineEdit->text().toInt();
  } else {
    throw std::runtime_error("Weird type width");
  }
}

template<>
uint32_t parseDataFromWidget<uint32_t>(const QWidget *widget) {
  const QLineEdit *lineEdit = dynamic_cast<const QLineEdit*>(widget);
  if (lineEdit == nullptr) {
    throw std::runtime_error("Parsing data from widget, uint32_t, but widget is not a lineedit");
  }
  if (std::is_same_v<uint, uint32_t>) {
    return lineEdit->text().toUInt();
  } else {
    throw std::runtime_error("Weird type width");
  }
}

template<>
int64_t parseDataFromWidget<int64_t>(const QWidget *widget) {
  const QLineEdit *lineEdit = dynamic_cast<const QLineEdit*>(widget);
  if (lineEdit == nullptr) {
    throw std::runtime_error("Parsing data from widget, int64_t, but widget is not a lineedit");
  }
  return lineEdit->text().toLongLong();
}

template<>
uint64_t parseDataFromWidget<uint64_t>(const QWidget *widget) {
  const QLineEdit *lineEdit = dynamic_cast<const QLineEdit*>(widget);
  if (lineEdit == nullptr) {
    throw std::runtime_error("Parsing data from widget, uint64_t, but widget is not a lineedit");
  }
  return lineEdit->text().toULongLong();
}

}

namespace protobuf_editor {
  
BuiltInTypeWidget::BuiltInTypeWidget(const pb::FieldDescriptor *fieldDescriptor, QWidget *parent) : ProtobufFieldWidget(fieldDescriptor, parent) {
  buildWidget();
}

void BuiltInTypeWidget::buildWidget() {
  if (fieldDescriptor_->type() == pb::FieldDescriptor::Type::TYPE_MESSAGE) {
    throw std::runtime_error("Built-in Type Widget was constructed with a field which is of type \"message\"");
  }


  // if (fieldDescriptor_->real_containing_oneof() != nullptr) {
  //   std::cout << "Not yet handling oneof-type \"" << fieldDescriptor_->full_name() << "\" for now" << std::endl;
  //   layout->addWidget(new QLabel(tr("[skipped] ")+QString::fromStdString(fieldDescriptor_->full_name())));
  //   return;
  // }
  
  // if (fieldDescriptor_->is_map()) {
  //   // Map is also "repeated", handle map first then move to next item
  //   std::cout << "Not yet handling map-type \"" << fieldDescriptor_->full_name() << "\" for now" << std::endl;
  //   layout->addWidget(new QLabel(tr("[skipped] ")+QString::fromStdString(fieldDescriptor_->full_name())));
  //   return;
  // }
  
  // if (fieldDescriptor_->is_repeated()) {
  //   std::cout << "Not yet handling repeated-type \"" << fieldDescriptor_->full_name() << "\" for now" << std::endl;
  //   layout->addWidget(new QLabel(tr("[skipped] ")+QString::fromStdString(fieldDescriptor_->full_name())));
  //   return;
  // }

  // if (fieldDescriptor_->type() == pb::FieldDescriptor::Type::TYPE_GROUP) {
  //   std::cout << "Not yet handling group-type \"" << fieldDescriptor_->full_name() << "\" for now" << std::endl;
  //   layout->addWidget(new QLabel(tr("[skipped] ")+QString::fromStdString(fieldDescriptor_->full_name())));
  //   return;
  // }

  fieldIsOptional_ = fieldDescriptor_->has_optional_keyword();

  // Create a label
  if (fieldIsOptional_) {
    labelWidget_ = new QCheckBox(QString::fromStdString(fieldDescriptor_->full_name()));
  } else {
    labelWidget_ = new QLabel(QString::fromStdString(fieldDescriptor_->full_name()));
  }

  if (fieldDescriptor_->type() == pb::FieldDescriptor::Type::TYPE_ENUM) {
    // Enums are a QComboBox widget
    QComboBox *comboBox = new QComboBox;
    dataWidget_ = comboBox;
    const pb::EnumDescriptor *enumDescriptor = fieldDescriptor_->enum_type();
    for (int enumValueIndex=0; enumValueIndex<enumDescriptor->value_count(); ++enumValueIndex) {
      const pb::EnumValueDescriptor *enumValueDescriptor = enumDescriptor->value(enumValueIndex);
      comboBox->addItem(QString::fromStdString(enumValueDescriptor->name()));
    }

    // Connect what will happen when we receive a new message
    connect(this, &BuiltInTypeWidget::messageSet, [this, comboBox](){
      if (message_ == nullptr) {
        throw std::runtime_error("Message was not set");
      }
      const pb::Reflection *reflection = message_->GetReflection();
      const bool isSet = fieldIsSet(reflection, fieldDescriptor_);
      if (fieldIsOptional_) {
        QCheckBox *labelAsCheckBox = dynamic_cast<QCheckBox*>(labelWidget_);
        if (labelAsCheckBox == nullptr) {
          throw std::runtime_error("Label for optional is not a checkbox");
        }
        labelAsCheckBox->setChecked(isSet);
      }
      if (!isSet) {
        // Nothing to do
        return;
      }

      // Field is set, get it from the message and update the combobox
      const pb::EnumValueDescriptor *enumValueDescriptor = reflection->GetEnum(*message_, fieldDescriptor_);
      // Note: This assumes that we added enum values into the combobox in order
      const int index = enumValueDescriptor->index();
      comboBox->setCurrentIndex(index);
    });

    // TODO: Connect what will happen when the combobox value is changed
  } else if (fieldDescriptor_->type() == pb::FieldDescriptor::Type::TYPE_BOOL) {
    // Booleans are a QCheckBox widget
    QCheckBox *checkBox = new QCheckBox;
    dataWidget_ = checkBox;
    
    // Connect what will happen when we receive a new message
    connect(this, &ProtobufEditorWidget::messageSet, [this, checkBox](){
      if (message_ == nullptr) {
        throw std::runtime_error("Message was not set");
      }
      const pb::Reflection *reflection = message_->GetReflection();
      bool isSet = fieldIsSet(reflection, fieldDescriptor_);
      if (fieldIsOptional_) {
        QCheckBox *labelAsCheckBox = dynamic_cast<QCheckBox*>(labelWidget_);
        if (labelAsCheckBox == nullptr) {
          throw std::runtime_error("Label for optional is not a checkbox");
        }
        labelAsCheckBox->setChecked(isSet);
      }
      if (!isSet) {
        // Nothing to do
        return;
      }

      // Field is set, get it from the message and update the checkbox
      const bool isTrue = reflection->GetBool(*message_, fieldDescriptor_);
      checkBox->setChecked(isTrue);
    });

    connect(checkBox, &QCheckBox::toggled, [this](){
      if (message_ == nullptr) {
        // No message to update, nothing to do
        return;
      }

      // TODO: Connect what will happen when the box is checked
    });
  } else {
    // All other built-in types are a QLineEdit widget
    QLineEdit *lineEdit = new QLineEdit;
    dataWidget_ = lineEdit;
    lineEdit->setMinimumWidth(200);

    // Connect what will happen when we receive a new message
    connect(this, &ProtobufEditorWidget::messageSet, [this, fieldDescriptor, labelWidget_, lineEdit](){
      if (message_ == nullptr) {
        throw std::runtime_error("Message was not set");
      }
      const pb::Reflection *reflection = message_->GetReflection();
      bool isSet = fieldIsSet(reflection, fieldDescriptor_);
      if (fieldIsOptional_) {
        QCheckBox *labelAsCheckBox = dynamic_cast<QCheckBox*>(labelWidget_);
        if (labelAsCheckBox == nullptr) {
          throw std::runtime_error("Label for optional is not a checkbox");
        }
        labelAsCheckBox->setChecked(isSet);
      }
      if (!isSet) {
        // Nothing to do
        return;
      }

      switch (fieldDescriptor_->type()) {
        case pb::FieldDescriptor::Type::TYPE_BYTES:
        case pb::FieldDescriptor::Type::TYPE_STRING: {
          const auto data = reflection->GetString(*message_, fieldDescriptor_);
          lineEdit->setText(QString::fromStdString(data));
          break;
        }
        case pb::FieldDescriptor::Type::TYPE_FLOAT: {
          const auto data = reflection->GetFloat(*message_, fieldDescriptor_);
          lineEdit->setText(QString::number(data));
          break;
        }
        case pb::FieldDescriptor::Type::TYPE_DOUBLE: {
          const auto data = reflection->GetDouble(*message_, fieldDescriptor_);
          lineEdit->setText(QString::number(data));
          break;
        }
        case pb::FieldDescriptor::Type::TYPE_INT32:
        case pb::FieldDescriptor::Type::TYPE_SINT32:
        case pb::FieldDescriptor::Type::TYPE_SFIXED32: {
          const auto data = reflection->GetInt32(*message_, fieldDescriptor_);
          lineEdit->setText(QString::number(data));
          break;
        }
        case pb::FieldDescriptor::Type::TYPE_UINT32:
        case pb::FieldDescriptor::Type::TYPE_FIXED32: {
          const auto data = reflection->GetUInt32(*message_, fieldDescriptor_);
          lineEdit->setText(QString::number(data));
          break;
        }
        case pb::FieldDescriptor::Type::TYPE_INT64:
        case pb::FieldDescriptor::Type::TYPE_SINT64:
        case pb::FieldDescriptor::Type::TYPE_SFIXED64: {
          const auto data = reflection->GetInt64(*message_, fieldDescriptor_);
          lineEdit->setText(QString::number(data));
          break;
        }
        case pb::FieldDescriptor::Type::TYPE_UINT64:
        case pb::FieldDescriptor::Type::TYPE_FIXED64: {
          const auto data = reflection->GetUInt64(*message_, fieldDescriptor_);
          lineEdit->setText(QString::number(data));
          break;
        }
        default:
          throw std::runtime_error("Unhandled type");
          break;
      }
    });
  }

  if (labelWidget_ == nullptr) {
    throw std::runtime_error("Didnt build a label widget");
  }

  if (dataWidget_ == nullptr) {
    throw std::runtime_error("Didnt build a data widget");
  }

  if (fieldIsOptional_) {
    // Connect the checkbox of the label to the enabled/disabled state of the data widget
    QCheckBox *labelAsCheckBox = dynamic_cast<QCheckBox*>(labelWidget_);
    if (labelAsCheckBox == nullptr) {
      throw std::runtime_error("Label for optional is not a checkbox");
    }
    connect(labelAsCheckBox, &QCheckBox::toggled, dataWidget_, &QWidget::setEnabled);
    dataWidget_->setEnabled(labelAsCheckBox->isChecked());

    // Update what happens when the label's checkbox is toggled
    connect(labelAsCheckBox, &QCheckBox::toggled, [this, fieldDescriptor, dataWidget_](bool checked){
      if (message_ == nullptr)  {
        // No message
        return;
      }
      const pb::Reflection *reflection = message_->GetReflection();
      if (!fieldIsOptional_) {
        throw std::runtime_error("Tried to set/unset optional, but this field isnt optional");
      }

      if (!checked) {
        // Box was unchecked, unset value
        reflection->ClearField(message_, fieldDescriptor_);
      } else {
        // Box was checked, set value (with the data that is in the widget)
        switch (fieldDescriptor_->type()) {
          case pb::FieldDescriptor::Type::TYPE_ENUM: {
            reflection->SetEnumValue(message_, fieldDescriptor_, parseEnumFromWidget(dataWidget_));
            break;
          }
          case pb::FieldDescriptor::Type::TYPE_BOOL: {
            reflection->SetBool(message_, fieldDescriptor_, parseBoolFromWidget(dataWidget_));
            break;
          }
          case pb::FieldDescriptor::Type::TYPE_BYTES:
          case pb::FieldDescriptor::Type::TYPE_STRING: {
            reflection->SetString(message_, fieldDescriptor_, parseDataFromWidget<std::string>(dataWidget_));
            break;
          }
          case pb::FieldDescriptor::Type::TYPE_FLOAT: {
            reflection->SetFloat(message_, fieldDescriptor_, parseDataFromWidget<float>(dataWidget_));
            break;
          }
          case pb::FieldDescriptor::Type::TYPE_DOUBLE: {
            reflection->SetDouble(message_, fieldDescriptor_, parseDataFromWidget<double>(dataWidget_));
            break;
          }
          case pb::FieldDescriptor::Type::TYPE_INT32:
          case pb::FieldDescriptor::Type::TYPE_SINT32:
          case pb::FieldDescriptor::Type::TYPE_SFIXED32: {
            reflection->SetInt32(message_, fieldDescriptor_, parseDataFromWidget<int32_t>(dataWidget_));
            break;
          }
          case pb::FieldDescriptor::Type::TYPE_UINT32:
          case pb::FieldDescriptor::Type::TYPE_FIXED32: {
            reflection->SetUInt32(message_, fieldDescriptor_, parseDataFromWidget<uint32_t>(dataWidget_));
            break;
          }
          case pb::FieldDescriptor::Type::TYPE_INT64:
          case pb::FieldDescriptor::Type::TYPE_SINT64:
          case pb::FieldDescriptor::Type::TYPE_SFIXED64: {
            reflection->SetInt64(message_, fieldDescriptor_, parseDataFromWidget<int64_t>(dataWidget_));
            break;
          }
          case pb::FieldDescriptor::Type::TYPE_UINT64:
          case pb::FieldDescriptor::Type::TYPE_FIXED64: {
            reflection->SetUInt64(message_, fieldDescriptor_, parseDataFromWidget<uint64_t>(dataWidget_));
            break;
          }
          default:
            throw std::runtime_error("Unhandled type");
            break;
        }
      }
    });
  }
  
  // Add the widgets to the layout
  QHBoxLayout *layout = new QHBoxLayout(this);
  layout->addWidget(labelWidget_);
  layout->addWidget(dataWidget_);
}

} // namespace protobuf_editor

namespace {

int parseEnumFromWidget(const QWidget *widget) {
  return {};
}

bool parseBoolFromWidget(const QWidget *widget) {
  const QCheckBox *checkBox = dynamic_cast<const QCheckBox*>(widget);
  if (checkBox == nullptr) {
    throw std::runtime_error("Parsing data from widget, bool, but widget is not a checkbox");
  }
  return checkBox->isChecked();
}

} // anonymous namespace