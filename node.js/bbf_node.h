#ifndef BBF_NODE_H
#define BBF_NODE_H

#include <node.h>

extern "C" {
  #include "ccv.h"
}

class Bbf : public node::ObjectWrap {
 public:
  static void Init(v8::Handle<v8::Object> target);

 private:
  Bbf();
  ~Bbf();

  static v8::Handle<v8::Value> New(const v8::Arguments& args);
  static v8::Handle<v8::Value> detect(const v8::Arguments& args);
  double counter_;
  ccv_bbf_classifier_cascade_t* cascade;
};

#endif
