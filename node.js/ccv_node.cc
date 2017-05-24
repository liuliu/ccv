#include <node.h>
#include <v8.h>
#include "bbf_node.h"


using namespace v8;

extern "C" {
  #include "ccv.h"
}

	
Handle<Value> Ccv_enable_default_cache(const Arguments& args) {
  HandleScope scope;

  ccv_enable_default_cache();

  return scope.Close(Null());
}


void init(Handle<Object> target) {
  Bbf::Init(target);

  target->Set(String::NewSymbol("ccv_enable_default_cache"), FunctionTemplate::New(Ccv_enable_default_cache)->GetFunction());
}
NODE_MODULE(ccv_node, init)
