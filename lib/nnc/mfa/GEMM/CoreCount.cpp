#include <IOKit/IOKitLib.h>
#include "../ccv_nnc_mfa.hpp"

#if TARGET_OS_MAC
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
#endif