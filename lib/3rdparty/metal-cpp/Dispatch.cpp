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
#include "Dispatch.hpp"



namespace Security {

//
// The base of the exception hierarchy.
//
CommonError::CommonError() : whatBuffer("CommonError")
{
}


//
// We strongly encourage catching all exceptions by const reference, so the copy
// constructor of our exceptions should never be called.
//
CommonError::CommonError(const CommonError &source)
{
    strlcpy(whatBuffer, source.whatBuffer, whatBufferSize);
}

CommonError::~CommonError() throw ()
{
}

void CommonError::LogBacktrace() {
  
}



//
// UnixError exceptions
//
UnixError::UnixError() : error(errno)
{
//    SECURITY_EXCEPTION_THROW_UNIX(this, errno);

    snprintf(whatBuffer, whatBufferSize, "UNIX errno exception: %d", this->error);
    printf("security_exception", "%s", what());
    LogBacktrace();
}

UnixError::UnixError(int err) : error(err)
{
//    SECURITY_EXCEPTION_THROW_UNIX(this, err);

    snprintf(whatBuffer, whatBufferSize, "UNIX error exception: %d", this->error);
    printf("security_exception", "%s", what());
    LogBacktrace();
}

const char *UnixError::what() const throw ()
{
    return whatBuffer;
}

int UnixError::unixError() const
{ return error; }

void UnixError::throwMe(int err) { throw UnixError(err); }

// @@@ This is a hack for the Network protocol state machine
UnixError UnixError::make(int err) { return UnixError(err); }



void ModuleNexusError::throwMe()
{
    throw ModuleNexusError();
}



//OSStatus ModuleNexusError::osStatus() const
//{
//    return errSecParam;
//}



int ModuleNexusError::unixError() const
{
    return EINVAL;
}


//
// The Error class thrown if Nexus operations fail
//
GlobalNexus::Error::~Error() throw()
{
}

void ModuleNexusCommon::do_create(void *(*make)())
{
    try
    {
        pointer = make();
    }
    catch (...)
    {
        pointer = NULL;
    }
}



void *ModuleNexusCommon::create(void *(*make)())
{
    dispatch_once(&once, ^{do_create(make);});
    
    if (pointer == NULL)
    {
        ModuleNexusError::throwMe();
    }
    
    return pointer;
}




//
// Mutex implementation
//
struct MutexAttributes {
  pthread_mutexattr_t recursive;
  pthread_mutexattr_t checking;
  
  MutexAttributes()
  {
    pthread_mutexattr_init(&recursive);
    pthread_mutexattr_settype(&recursive, PTHREAD_MUTEX_RECURSIVE);
#if !defined(NDEBUG)
    pthread_mutexattr_init(&checking);
    pthread_mutexattr_settype(&checking, PTHREAD_MUTEX_ERRORCHECK);
#endif //NDEBUG
  }
};


static ModuleNexus<MutexAttributes> mutexAttrs;


Mutex::Mutex()
{
  check(pthread_mutex_init(&me, NULL));
}

Mutex::Mutex(Type type)
{
  switch (type) {
  case normal:
    check(pthread_mutex_init(&me, IFELSEDEBUG(&mutexAttrs().checking, NULL)));
    break;
  case recursive:    // requested recursive (is also checking, always)
    check(pthread_mutex_init(&me, &mutexAttrs().recursive));
    break;
  };
}


Mutex::~Mutex()
{
    int result = pthread_mutex_destroy(&me);
    if(result) {
      printf("Probable bug: error destroying Mutex: %d", result);
    }
  check(result);
}


void Mutex::lock()
{
  check(pthread_mutex_lock(&me));
}


bool Mutex::tryLock()
{
  if (int err = pthread_mutex_trylock(&me)) {
    if (err != EBUSY)
      UnixError::throwMe(err);
    return false;
  }

  return true;
}


void Mutex::unlock()
{
    int result = pthread_mutex_unlock(&me);
  check(result);
}

} // end namespace Security



namespace Dispatch {

ExceptionAwareEnqueuing::ExceptionAwareEnqueuing()
: mExceptionPending(false)
{ }

void ExceptionAwareEnqueuing::enqueueWithDispatcher(void (^dispatcher)(dispatch_block_t), dispatch_block_t block)
{
  if (mExceptionPending)
    return;

  dispatcher(^{
    if (mExceptionPending)
      return;
    try {
      block();
    } catch (...) {
      Security::StLock<Security::Mutex> _(mLock);
      mExceptionPending = true;
      mException = std::current_exception();
    }
  });
}

void ExceptionAwareEnqueuing::throwPendingException()
{
  if (mExceptionPending) {
    mExceptionPending = false;
    std::rethrow_exception(mException);
  }
}



Queue::Queue(const char *label, bool concurrent, dispatch_qos_class_t qos_class)
{
  dispatch_queue_attr_t attr = concurrent ? DISPATCH_QUEUE_CONCURRENT : DISPATCH_QUEUE_SERIAL;
  attr = dispatch_queue_attr_make_with_qos_class(attr, qos_class, 0);
  mQueue = dispatch_queue_create(label, attr);
}

Queue::~Queue()
{
  dispatch_barrier_sync(mQueue, ^{});
  dispatch_release(mQueue);
}

void Queue::enqueue(dispatch_block_t block)
{
  enqueuing.enqueueWithDispatcher(^(dispatch_block_t block){ dispatch_async(mQueue, block); }, block);
}

void Queue::wait()
{
  dispatch_barrier_sync(mQueue, ^{});
  enqueuing.throwPendingException();
}



Group::Group()
{
  mGroup = dispatch_group_create();
}

Group::~Group()
{
  dispatch_group_wait(mGroup, DISPATCH_TIME_FOREVER);
  dispatch_release(mGroup);
}

void Group::enqueue(dispatch_queue_t queue, dispatch_block_t block)
{
  enqueuing.enqueueWithDispatcher(^(dispatch_block_t block){ dispatch_group_async(mGroup, queue, block); }, block);
}

void Group::wait()
{
  dispatch_group_wait(mGroup, DISPATCH_TIME_FOREVER);
  enqueuing.throwPendingException();
}



Semaphore::Semaphore(long count) {
  mSemaphore = dispatch_semaphore_create(count);
}

Semaphore::Semaphore(Semaphore& semaphore)
: mSemaphore(semaphore.mSemaphore)
{
  dispatch_retain(mSemaphore);
}

Semaphore::~Semaphore() {
  dispatch_release(mSemaphore);
}

bool Semaphore::signal() {
  return dispatch_semaphore_signal(mSemaphore) == 0;
}

bool Semaphore::wait(dispatch_time_t timeout) {
  return dispatch_semaphore_wait(mSemaphore, timeout) == 0;
}


// Transfer ownership of held resource.
SemaphoreWait::SemaphoreWait(SemaphoreWait &originalWait)
: mSemaphore(originalWait.mSemaphore), mAcquired(originalWait.mAcquired)
{
  originalWait.mAcquired = false;
}

SemaphoreWait::SemaphoreWait(Semaphore &semaphore, dispatch_time_t timeout)
: mSemaphore(semaphore)
{
  mAcquired = mSemaphore.wait(timeout);
}

SemaphoreWait::~SemaphoreWait()
{
  if (mAcquired)
    mSemaphore.signal();
}


} // end namespace Dispatch
