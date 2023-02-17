#include "protobufFieldWidget.hpp"

namespace pb = google::protobuf;

namespace protobuf_editor {

ProtobufFieldWidget::ProtobufFieldWidget(const google::protobuf::FieldDescriptor *fieldDescriptor, QWidget *parent) : QWidget(parent), fieldDescriptor_(fieldDescriptor) {
  if (fieldDescriptor_ == nullptr) {
    // No field descriptor, must be a top-level, thus is not optional and is always set
    fieldIsOptional_ = false;
  } else {
    fieldIsOptional_ = fieldDescriptor_->has_optional_keyword();
  }
}
ProtobufFieldWidget::~ProtobufFieldWidget() {}

void ProtobufFieldWidget::setMessage(pb::Message *msg) {
  message_ = msg;
  emit messageSet();
}

bool ProtobufFieldWidget::fieldIsSet(const pb::Reflection *reflection) const {
  if (fieldDescriptor_ == nullptr) {
    // No field descriptor, must be a top-level, thus is not optional and is always set
    return true;
  }
  if (!fieldIsOptional_) {
    // Non-optional fields are always set
    return true;
  }
  return reflection->HasField(*message_, fieldDescriptor_);
}

} // namespace protobuf_editor