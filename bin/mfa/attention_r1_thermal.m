#import <Foundation/Foundation.h>
#include <stdio.h>
int main(void) { @autoreleasepool { printf("%ld\n", (long)[[NSProcessInfo processInfo] thermalState]); } return 0; }
