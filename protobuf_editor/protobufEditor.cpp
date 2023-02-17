#include "protobufEditor.hpp"
#include "messageTypeWidget.hpp"

#include "proto/test.pb.h"

#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QScrollArea>
#include <QLineEdit>
#include <QVBoxLayout>

#include <iostream>

namespace pb = google::protobuf;

ProtobufEditor::ProtobufEditor(QWidget *parent) : QWidget{parent} {
  QVBoxLayout *layout = new QVBoxLayout(this);
  QScrollArea *scrollArea = new QScrollArea;

  const pb::Descriptor *desc = proto::test::Test::GetDescriptor();
  MessageTypeWidget *w = new MessageTypeWidget(*desc);
  scrollArea->setWidget(w);
  layout->addWidget(scrollArea);

  // Never deallocate, want widget to never reference something which does not exist
  proto::test::Test *testMsg = new proto::test::Test;
  {
    // Fill with some fake data
    testMsg->set_opt_f(69.420);
    auto *o_n = testMsg->mutable_opt_nested();
    o_n->set_data("Yo momma");

  }
  w->setMessage(testMsg);
  // connect(w, &MessageTypeWidget::messageUpdated, /* whoever cares*/);
}
