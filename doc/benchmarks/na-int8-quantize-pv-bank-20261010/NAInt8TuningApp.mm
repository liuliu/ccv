#import <Foundation/Foundation.h>
#include <os/proc.h>
#include <cstdio>
int na_int8_tuning_run(int argc, char** argv);

int main(int argc, char* argv[])
{
  @autoreleasepool {
    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
    printf("thermal_state=%ld os=%s\n", (long)[NSProcessInfo processInfo].thermalState,
        [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
    printf("available_memory_bytes=%llu\n", (unsigned long long)os_proc_available_memory());
    const int result = na_int8_tuning_run(argc, argv);
    printf("probe_exit=%d thermal_end=%ld\n", result, (long)[NSProcessInfo processInfo].thermalState);
    fflush(stdout);
    fflush(stderr);
    return result;
  }
}
