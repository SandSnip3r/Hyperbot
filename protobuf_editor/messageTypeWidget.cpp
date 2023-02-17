#include "builtInTypeWidget.hpp"
#include "messageTypeWidget.hpp"

#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QCheckBox>
#include <QLineEdit>
#include <QComboBox>

namespace pb = google::protobuf;

namespace protobuf_editor {

MessageTypeWidget::MessageTypeWidget(const pb::Descriptor *descriptor, const pb::FieldDescriptor *fieldDescriptor, QWidget *parent) : ProtobufFieldWidget(fieldDescriptor, parent), descriptor_(descriptor) {
  buildWidget();
}

void MessageTypeWidget::setMessage(pb::Message *msg) {
  message_ = msg;

  // TODO: If we're optional, set enbled/disabled based on message data

  for (auto *nested : nestedWidgets_) {
    if (nested == nullptr) {
      // TODO: Throw here once we handle all field types
      std::cout << "Warning! Nested widget is null. This is ok now since we skip certain pb field types" << std::endl;
      continue;
    }
    nested->setMessage(message_);
  }
  emit messageSet();
}

void MessageTypeWidget::buildWidget() {
  // Create a layout (LayoutA) for our entire widget
  QVBoxLayout *overallLayout = new QVBoxLayout(this);
  // Specify no margin for this layout, since we want it to look like the QGroupBox is what we are
  overallLayout->setContentsMargins(0,0,0,0);
  // TODO: Ideally we'd inherit from QGroupBox, but that seems tricky given that we've already inherited from something which inherits from QObject

  // Create a groupbox for this message
  QGroupBox *groupBox;
  if (fieldDescriptor_ != nullptr) {
    // We are a nested message
    groupBox = new QGroupBox(QString::fromStdString(fieldDescriptor->full_name()));
  } else {
    // We are a root-level message
    // TODO: Dont create a group box, instead just create a QWidget
    groupBox = new QGroupBox(tr("Root-level Message"));
  }
  
  // Put the groupbox (and only that groupbox) into the layout LayoutA
  overallLayout->addWidget(groupBox);
  
  // Create a layout (LayoutB) inside the groupbox
  QVBoxLayout *groupBoxLayout = new QVBoxLayout(groupBox);

  // Handle if we are optional
  if (fieldIsOptional_) {
    groupBox->setCheckable(true);
    connect(groupBox, &QGroupBox::toggled, this, &QWidget::setEnabled);
    groupBox->setChecked(false);
  }

  // Iterate over all fields, create widgets for them, and add them to our layout
  for (int fieldIndex=0; fieldIndex<descriptor_->field_count(); ++fieldIndex) {
    const pb::FieldDescriptor *fieldDescriptor = descriptor_->field(fieldIndex);

    // Skip unhandled types for now
    if (fieldDescriptor->real_containing_oneof() != nullptr) {
      std::cout << "Skipping oneof \"" << fieldDescriptor->full_name() << "\" for now" << std::endl;
      groupBoxLayout->addWidget(new QLabel(tr("[skipped] ")+QString::fromStdString(fieldDescriptor->full_name())));
      nestedWidgets_.push_back(nullptr); // TODO: no
      continue;
    }
    
    if (fieldDescriptor->is_map()) {
      // Map is also "repeated", handle map first then move to next item
      std::cout << "Skipping map \"" << fieldDescriptor->full_name() << "\" for now" << std::endl;
      groupBoxLayout->addWidget(new QLabel(tr("[skipped] ")+QString::fromStdString(fieldDescriptor->full_name())));
      nestedWidgets_.push_back(nullptr); // TODO: no
      continue;
    }
    
    if (fieldDescriptor->is_repeated()) {
      std::cout << "Skipping repeated \"" << fieldDescriptor->full_name() << "\" for now" << std::endl;
      groupBoxLayout->addWidget(new QLabel(tr("[skipped] ")+QString::fromStdString(fieldDescriptor->full_name())));
      nestedWidgets_.push_back(nullptr); // TODO: no
      continue;
    }

    if (fieldDescriptor->type() == pb::FieldDescriptor::Type::TYPE_GROUP) {
      std::cout << "Skipping group \"" << fieldDescriptor->full_name() << "\" for now" << std::endl;
      groupBoxLayout->addWidget(new QLabel(tr("[skipped] ")+QString::fromStdString(fieldDescriptor->full_name())));
      nestedWidgets_.push_back(nullptr); // TODO: no
      continue;
    }
    
    ProtobufFieldWidget *widgetForField;
    if (fieldDescriptor->type() == pb::FieldDescriptor::Type::TYPE_MESSAGE) {
      // Is a nested message type
      const pb::Descriptor *nestedDescriptor = fieldDescriptor->message_type();
      if (nestedDescriptor == nullptr) {
          throw std::runtime_error("Nested field descriptor is null");
      }
      widgetForField = new MessageTypeWidget(nestedDescriptor, fieldDescriptor);
    } else {
      // Is a built-in type
      widgetForField = new BuiltInTypeWidget(fieldDescriptor);
    }
    groupBoxLayout->addWidget(widgetForField);
    nestedWidgets_.push_back(widgetForField);
  }

  // Connect what will happen when we received a new message
  connect(this, &MessageTypeWidget::messageSet, [this, fieldDescriptor, groupBox, nestedWidget](){
    if (message_ == nullptr) {
      throw std::runtime_error("Message was not set");
    }
    const pb::Reflection *reflection = message_->GetReflection();
    const bool isSet = fieldIsSet(reflection, fieldDescriptor);
    if (fieldIsOptional_) {
      if (!groupBox->isCheckable()) {
        throw std::runtime_error("Group box is not checkable for optional field");
      }
      if (!reflection->HasField(*message_, fieldDescriptor)) {
        isSet = false;
      }
      groupBox->setChecked(isSet);
    }
  });

  // TODO: Connect what will happen when the message is updated
  // connect(labelAsCheckBox, &QCheckBox::toggled, [this, fieldDescriptor](bool checked){
  //   if (message_ == nullptr)  {
  //     // No message
  //     return;
  //   }
  //   const pb::Reflection *reflection = message_->GetReflection();
  //   if (!fieldDescriptor->has_optional_keyword()) {
  //     throw std::runtime_error("Tried to set/unset optional, but this field isnt optional");
  //   }
  //   // TODO: Set to the value from the widget
  //   if (checked) {
  //     switch (fieldDescriptor->type()) {
  //       case pb::FieldDescriptor::Type::TYPE_MESSAGE: {
  //         pb::Message *msg = reflection->MutableMessage(message_, fieldDescriptor);
  //         (void)msg;
  //         break;
  //       }

  // ==============================================================================
}

} // namespace protobuf_editor