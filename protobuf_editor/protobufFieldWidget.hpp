#ifndef PROTOBUF_EDITOR_PROTOBUF_FIELD_WIDGET_HPP_
#define PROTOBUF_EDITOR_PROTOBUF_FIELD_WIDGET_HPP_

#include <google/protobuf/message.h>

#include <QWidget>

namespace protobuf_editor {

class ProtobufFieldWidget : public QWidget {
  Q_OBJECT
public:
  explicit ProtobufFieldWidget(const google::protobuf::FieldDescriptor *fieldDescriptor=nullptr, QWidget *parent=nullptr);
  virtual void setMessage(google::protobuf::Message *msg);
  virtual ~ProtobufFieldWidget() = 0;
protected:
  const google::protobuf::FieldDescriptor* const fieldDescriptor_;
  bool fieldIsOptional_;
  google::protobuf::Message *message_{nullptr};
  bool fieldIsSet(const google::protobuf::Reflection *reflection) const;
signals:
  void messageSet();
};

} // namespace protobuf_editor

#endif // PROTOBUF_EDITOR_PROTOBUF_FIELD_WIDGET_HPP_
