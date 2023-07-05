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
} // end namespace Security
