#ifndef GEMMHeaders_hpp
#define GEMMHeaders_hpp

#include <string>

/// Create the source code for the 'metal\_simdgroup\_event' header.
///
/// I may have found the hardware bug with async copies on M1. If you shoot
/// off an async copy, you need to read from its contents later in the
/// the shader. Otherwise, something inside the hardware (like a
/// DispatchSemaphore) will be waiting indefinitely to be notified. The bug
/// is a bit flaky, and only shows up for certain problem configurations. The
/// side effects are catastrophic; the GPU might freeze up until the computer
/// reboots.
///
/// Workaround: if an async copy from device -> threadgroup is launched,
/// guarantee that both:
/// - The threadgroup will enter another `threadgroup_barrier` before the end of
///   the kernel.
/// - The results of the async copy will be read from. This means at least one
///   thread must dereference a pointer within the region of threadgroup memory.
std::string createMetalSimdgroupEvent();

/// Create the source code for the 'metal\_simdgroup\_matrix\_storage' header.
std::string createMetalSimdgroupMatrixStorage();

#endif /* GEMMHeaders_hpp */
