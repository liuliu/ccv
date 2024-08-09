#include "../ccv_nnc_mfa.hpp"

#if TARGET_OS_MAC

// Apple requires apps to detail why they use IOKit on the Mac App Store.
// Therefore, IOKit is not an option for certain clients. As an alternative,
// OpenCL can be used.
//
// All of these issues only apply to macOS. On iOS, neither IOKit nor OpenCL
// are available. However, the variation in core count is small enough to
// provide accurate order-of-magnitude estimates through other means.
#if 0
#include <IOKit/IOKitLib.h>
int64_t findCoreCount() {
  // Create a matching dictionary with "AGXAccelerator" class name
  auto matchingDict = IOServiceMatching("AGXAccelerator");
  
  // Get an iterator for matching services
  io_iterator_t iterator = 0;
  {
    kern_return_t io_registry_error =
    IOServiceGetMatchingServices(kIOMainPortDefault, matchingDict, &iterator);
    CCV_NNC_MFA_PRECONDITION(io_registry_error == 0);
  }
  
  // Get the first (and only) GPU entry from the iterator
  io_iterator_t gpuEntry = IOIteratorNext(iterator);
  
  // Check if the entry is valid
  CCV_NNC_MFA_PRECONDITION(gpuEntry != MACH_PORT_NULL);
  
  // Release the iterator
  IOObjectRelease(iterator);
  
  // Get the "gpu-core-count" property from gpuEntry
  std::string key = "gpu-core-count";
  IOOptionBits options = 0; // No options needed
  CFStringRef keyRef =
  CFStringCreateWithCString(kCFAllocatorDefault, key.c_str(), kCFStringEncodingUTF8);
  CFTypeRef gpuCoreCount =
  IORegistryEntrySearchCFProperty(gpuEntry, kIOServicePlane, keyRef, nil, options);
  
  // Check if the property is valid
  CCV_NNC_MFA_PRECONDITION(gpuCoreCount != nil);
  
  // Cast the property to CFNumberRef
  CFNumberRef gpuCoreCountNumber = (CFNumberRef)gpuCoreCount;
  
  // Check if the number type is sInt64
  CFNumberType type = CFNumberGetType(gpuCoreCountNumber);
  CCV_NNC_MFA_PRECONDITION(type == kCFNumberSInt64Type);
  
  // Get the value of the number as Int64
  int64_t value = 0;
  bool result = CFNumberGetValue(gpuCoreCountNumber, type, &value);
  
  // Check for errors
  CCV_NNC_MFA_PRECONDITION(result != false);
  
  return value;
}
#else

// #include <OpenCL/opencl.h>

// First-call latency on macOS: 44,000 microseconds
// Amortized latency on macOS: 0 microseconds
int64_t findCoreCount() {
/*
  cl_platform_id platformID;
  uint32_t retNumPlatforms;
  int32_t ret = clGetPlatformIDs(1, &platformID, &retNumPlatforms);
  CCV_NNC_MFA_PRECONDITION(ret == 0);
  CCV_NNC_MFA_PRECONDITION(retNumPlatforms == 1);
  
  cl_device_id deviceID;
  uint32_t retNumDevices;
  ret = clGetDeviceIDs
  (platformID, uint64_t(CL_DEVICE_TYPE_DEFAULT), 1, &deviceID, &retNumDevices);
  CCV_NNC_MFA_PRECONDITION(ret == 0);
  CCV_NNC_MFA_PRECONDITION(retNumDevices == 1);
  
  uint32_t paramValue;
  ret = clGetDeviceInfo
  (deviceID, uint64_t(CL_DEVICE_MAX_COMPUTE_UNITS), 4, &paramValue, nil);
*/
  return 18;
}
#endif

#endif
