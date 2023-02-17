#ifndef PROTOBUF_EDITOR_BUILT_IN_TYPE_WIDGET_HPP_
#define PROTOBUF_EDITOR_BUILT_IN_TYPE_WIDGET_HPP_

#include "protobufFieldWidget.hpp"

#include <google/protobuf/message.h>

#include <QWidget>

namespace protobuf_editor {

class BuiltInTypeWidget : public ProtobufFieldWidget {
  Q_OBJECT
public:
  explicit BuiltInTypeWidget(const google::protobuf::FieldDescriptor *fieldDescriptor, QWidget *parent=nullptr);
private:
  QWidget *labelWidget_{nullptr};
  QWidget *dataWidget_{nullptr};

  void buildWidget();
};

} // namespace protobuf_editor

#endif // PROTOBUF_EDITOR_BUILT_IN_TYPE_WIDGET_HPP_
