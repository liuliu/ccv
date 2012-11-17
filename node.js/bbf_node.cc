#include <node.h>
#include "bbf_node.h"

using namespace v8;

Bbf::Bbf() {};
Bbf::~Bbf() {};

void Bbf::Init(Handle<Object> target) {
  // Prepare constructor template
  Local<FunctionTemplate> tpl = FunctionTemplate::New(New);
  tpl->SetClassName(String::NewSymbol("Bbf"));
  tpl->InstanceTemplate()->SetInternalFieldCount(1);
  // Prototype
  tpl->PrototypeTemplate()->Set(String::NewSymbol("detect"),
      FunctionTemplate::New(detect)->GetFunction());

  Persistent<Function> constructor = Persistent<Function>::New(tpl->GetFunction());
  target->Set(String::NewSymbol("bbf"), constructor);
}

Handle<Value> Bbf::New(const Arguments& args) {
  HandleScope scope;

  Bbf* obj = new Bbf();

  Local<v8::String> classifierPath = args[0]->ToString();
  char *classifier_path_cstring = (char *)malloc(sizeof(char *) * classifierPath->Length());
  classifierPath->WriteAscii(classifier_path_cstring);

  obj->cascade = ccv_load_bbf_classifier_cascade(classifier_path_cstring);
  obj->Wrap(args.This());

  free(classifier_path_cstring);

  return args.This();
}

Handle<Value> Bbf::detect(const Arguments& args) {
  HandleScope scope;

  Local<v8::String> imagePath = args[0]->ToString();
  char *image_path_cstring = (char *)malloc(sizeof(char *) * imagePath->Length());
  imagePath->WriteAscii(image_path_cstring);

  Bbf* obj = ObjectWrap::Unwrap<Bbf>(args.This());
  obj->counter_ += 1;

  ccv_dense_matrix_t* image = 0;
  ccv_read(image_path_cstring, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
  free(image_path_cstring);

  if (image != 0) {
    ccv_array_t* seq = ccv_bbf_detect_objects(image, &obj->cascade, 1, ccv_bbf_default_params);

    v8::Local<v8::Array> matches = v8::Array::New(seq->rnum);

    for (int i = 0; i < seq->rnum; i++) {
      ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq, i);

      Local<Object> match = Object::New();
      match->Set(String::NewSymbol("x"), v8::Number::New(comp->rect.x));
      match->Set(String::NewSymbol("y"), v8::Number::New(comp->rect.y));
      match->Set(String::NewSymbol("width"), v8::Number::New(comp->rect.width));
      match->Set(String::NewSymbol("height"), v8::Number::New(comp->rect.height));
      match->Set(String::NewSymbol("confidence"), v8::Number::New(comp->confidence));

      matches->Set(v8::Number::New(i), match);
    }
    ccv_array_free(seq);
    ccv_matrix_free(image);
    ccv_disable_cache();

    return scope.Close(matches);
  }

  return scope.Close(Null());
}
