#include "NARegisterAttentionKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

NARegisterAttentionKernel::NARegisterAttentionKernel(
    NARegisterAttentionKernelDescriptor descriptor, MTL::Device* device) {
  CCV_NNC_MFA_PRECONDITION(descriptor.D == 128 || descriptor.D == 256);
  std::string source =
#include "../3rdparty/mlx/attention.metal.inc"
  ;
  // Four groups own 16 query rows each. Wide heads split each row group into
  // two 128-column output slices, reducing live accumulators per SIMD group.
  const std::string kernel = descriptor.D == 256 ?
      "attention_nax_dsplit<half,64,32,256,4,2,float,float>" :
      "attention_nax<half,64,32,128,128,4,1,float,float>";
  source += "\ntemplate [[host_name(\"attention_register\")]] kernel decltype(" +
      kernel + ") " + kernel + ";\n";
  NS::Error* error = nullptr;
  library = NS::TransferPtr(device->newLibrary(
      NS::String::string(source.c_str(), NS::UTF8StringEncoding), nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}
