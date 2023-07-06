/*
 * Copyright (c) 2014 Apple Inc. All Rights Reserved.
 *
 * @APPLE_LICENSE_HEADER_START@
 *
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 *
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 *
 * @APPLE_LICENSE_HEADER_END@
 */

//
// dispatch - libdispatch wrapper
//
#ifndef _H_DISPATCH
#define _H_DISPATCH

#include <AvailabilityMacros.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <libkern/OSAtomic.h>
#include <pthread.h>
#include <stdio.h>

#include <exception>



namespace Security {

//
// Elementary debugging support.
// #include <debugging.h> for more debugging facilities.
//
#define IFDEBUG(it)    IFELSEDEBUG(it,)
#define IFNDEBUG(it)  IFELSEDEBUG(,it)

#if defined(NDEBUG)

# define safe_cast  static_cast
# define safer_cast  static_cast

# define IFELSEDEBUG(d,nd) nd

#else

template <class Derived, class Base>
inline Derived safer_cast(Base &base)
{
    return dynamic_cast<Derived>(base);
}

template <class Derived, class Base>
inline Derived safe_cast(Base *base)
{
    if (base == NULL)
        return NULL;  // okay to cast NULL to NULL
    Derived p = dynamic_cast<Derived>(base);
    assert(p);
    return p;
}

# define IFELSEDEBUG(d,nd) d

#endif //NDEBUG

//
// Place this into your class definition if you don't want it to be copyable
// or asignable. This will not prohibit allocation on the stack or in static
// memory, but it will make anything derived from it, and anything containing
// it, fixed-once-created. A proper object, I suppose.
//
#define NOCOPY(Type)  \
  private: Type(const Type &) DEPRECATED_IN_MAC_OS_X_VERSION_10_0_AND_LATER; \
  void operator = (const Type &) DEPRECATED_IN_MAC_OS_X_VERSION_10_0_AND_LATER;
  
//
// Common base of Security exceptions that represent error conditions.
// All can yield Unix or OSStatus error codes as needed, though *how*
// is up to the subclass implementation.
// CSSM_RETURN conversions are done externally in (???).
//
class CommonError : public std::exception {
protected:
    CommonError();
    CommonError(const CommonError &source);
public:
    virtual ~CommonError() throw ();

//    virtual OSStatus osStatus() const = 0;
    virtual int unixError() const = 0;

    char whatBuffer[128];
    const size_t whatBufferSize = sizeof(whatBuffer);

    static void LogBacktrace();
};


//
// Genuine Unix-originated errors identified by an errno value.
// This includes secondary sources such as pthreads.
//
class UnixError : public CommonError {
protected:
    UnixError();
    UnixError(int err);
public:
    const int error;
//    virtual OSStatus osStatus() const;
    virtual int unixError() const;
    virtual const char *what () const throw ();
    
    static void check(int result)    { if (result == -1) throwMe(); }
    static void throwMe(int err = errno) __attribute__((noreturn));

    // @@@ This is a hack for the Network protocol state machine
    static UnixError make(int err = errno) DEPRECATED_ATTRIBUTE;
};

// Something that gets thrown when ModuleNexus creation fails
class ModuleNexusError : public CommonError {
protected:
    ModuleNexusError() {}

public:
//    virtual OSStatus osStatus() const;
    virtual int unixError() const;
    static void throwMe() __attribute__((noreturn));
};



//
// GlobalNexus is the common superclass of all globality scopes.
// A Nexus is an *access point* to the *single* object of a given
// type in the Nexus's particular scope.
//
class GlobalNexus {
public:
    class Error : public std::exception {
    public:
        virtual ~Error() throw();
        const char * const message;
        Error(const char *m) : message(m) { }
        const char *what() const throw() { return message; }
    };
};


class ModuleNexusCommon : public GlobalNexus {
private:
    void do_create(void *(*make)());

protected:
    void *create(void *(*make)());
    void lock() {OSSpinLockLock(&access);}
    void unlock() {OSSpinLockUnlock(&access);}

protected:
    // all of these will be statically initialized to zero
  void *pointer;
    dispatch_once_t once;
    OSSpinLock access;
};

template <class Type>
class ModuleNexus : public ModuleNexusCommon {
public:
    Type &operator () ()
    {
        lock();
        
        try
        {
            if (pointer == NULL)
            {
                pointer = create(make);
            }
            
            unlock();
        }
        catch (...)
        {
            unlock();
            throw;
        }
        
    return *reinterpret_cast<Type *>(pointer);
    }
  
  // does the object DEFINITELY exist already?
  bool exists() const
  {
        bool result;
        lock();
        result = pointer != NULL;
        unlock();
        return result;
  }
    
  // destroy the object (if any) and start over - not really thread-safe
    void reset()
    {
        lock();
        if (pointer != NULL)
        {
            delete reinterpret_cast<Type *>(pointer);
            pointer = NULL;
            once = 0;
        }
        unlock();
    }
    
private:
    static void *make() { return new Type; }
};



//
// Pthread Synchronization primitives.
// These have a common header, strictly for our convenience.
//
class LockingPrimitive {
protected:
  LockingPrimitive() { }
  
    void check(int err)  { if (err) UnixError::throwMe(err); }
};
  
//
// Mutexi
//
class Mutex : public LockingPrimitive {
    NOCOPY(Mutex)
    friend class Condition;

public:
  enum Type {
    normal,
    recursive
  };
  
    Mutex();              // normal
  Mutex(Type type);          // recursive
  ~Mutex();              // destroy (must be unlocked)
    void lock();            // lock and wait
  bool tryLock();            // instantaneous lock (return false if busy)
    void unlock();            // unlock (must be locked)

private:
    pthread_mutex_t me;
};
  
//
// A guaranteed-unlocker stack-based class.
// By default, this will use lock/unlock methods, but you can provide your own
// alternates (to, e.g., use enter/exit, or some more specialized pair of operations).
//
// NOTE: StLock itself is not thread-safe. It is intended for use (usually on the stack)
// by a single thread.
//
template <class Lock,
  void (Lock::*_lock)() = &Lock::lock,
  void (Lock::*_unlock)() = &Lock::unlock>
class StLock {
public:
  StLock(Lock &lck) : me(lck)      { (me.*_lock)(); mActive = true; }
  StLock(Lock &lck, bool option) : me(lck), mActive(option) { }
  ~StLock()              { if (mActive) (me.*_unlock)(); }

  bool isActive() const        { return mActive; }
  void lock()              { if(!mActive) { (me.*_lock)(); mActive = true; }}
  void unlock()            { if(mActive) { (me.*_unlock)(); mActive = false; }}
  void release()            { assert(mActive); mActive = false; }

  operator const Lock &() const    { return me; }
  
protected:
  Lock &me;
  bool mActive;
};

} // end namespace Security



namespace Dispatch {


// Wraps dispatch objects which can be used to queue blocks, i.e. dispatch groups and queues.
// If a block throws an exception, no further blocks are enqueued and the exception is rethrown
// after waiting for completion of all blocks.
class ExceptionAwareEnqueuing {
  NOCOPY(ExceptionAwareEnqueuing)
public:
  ExceptionAwareEnqueuing();

  void enqueueWithDispatcher(void (^dispatcher)(dispatch_block_t), dispatch_block_t block);
  void throwPendingException();
private:
  Security::Mutex mLock;
  bool mExceptionPending;
  std::exception_ptr mException;
};


class Queue {
  NOCOPY(Queue)
public:
  Queue(const char *label, bool concurrent, dispatch_qos_class_t qos_class = QOS_CLASS_DEFAULT);
  virtual ~Queue();

  operator dispatch_queue_t () const { return mQueue; }

  void enqueue(dispatch_block_t block);
  void wait();

private:
  ExceptionAwareEnqueuing enqueuing;
  dispatch_queue_t mQueue;
};


class Group {
  NOCOPY(Group)
public:
  Group();
  virtual ~Group();

  operator dispatch_group_t () const { return mGroup; }

  void enqueue(dispatch_queue_t queue, dispatch_block_t block);
  void wait();

private:
  ExceptionAwareEnqueuing enqueuing;
  dispatch_group_t mGroup;
};


class Semaphore {
  NOCOPY(Semaphore)
public:
  Semaphore(long count);
  Semaphore(Semaphore& semaphore);
  virtual ~Semaphore();

  operator dispatch_semaphore_t () const { return mSemaphore; };

  bool signal();
  bool wait(dispatch_time_t timeout = DISPATCH_TIME_FOREVER);

private:
  dispatch_semaphore_t mSemaphore;
};


class SemaphoreWait {
  NOCOPY(SemaphoreWait)
public:
  SemaphoreWait(SemaphoreWait& originalWait);
  SemaphoreWait(Semaphore& semaphore, dispatch_time_t timeout = DISPATCH_TIME_FOREVER);
  virtual ~SemaphoreWait();

  bool acquired() const { return mAcquired; };

private:
  Semaphore &mSemaphore;
  bool mAcquired;
};



} // end namespace Dispatch



#endif // !_H_DISPATCH
