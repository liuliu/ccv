/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include "3rdparty/thrust/detail/config.h"
#include "3rdparty/thrust/detail/functional/actor.h"
#include "3rdparty/thrust/detail/functional/composite.h"
#include "3rdparty/thrust/detail/functional/operators/operator_adaptors.h"
#include "3rdparty/thrust/functional.h"

namespace thrust
{

// XXX WAR circular inclusion with this forward declaration
template<typename,typename,typename> struct binary_function;

namespace detail
{
namespace functional
{

// XXX WAR circular inclusion with this forward declaration
template<typename> struct as_actor;

// there's no standard assign functional, so roll an ad hoc one here
template<typename T>
  struct assign
    : thrust::binary_function<T&,T,T&>
{
  __host__ __device__ T& operator()(T &lhs, const T &rhs) const { return lhs = rhs; }
}; // end assign

template<typename Eval, typename T>
  struct assign_result
{
  typedef actor<
    composite<
      binary_operator<assign>,
      actor<Eval>,
      typename as_actor<T>::type
    >
  > type;
}; // end assign_result

template<typename Eval, typename T>
  __host__ __device__
    typename assign_result<Eval,T>::type
      do_assign(const actor<Eval> &_1, const T &_2)
{
  return compose(binary_operator<assign>(),
                 _1,
                 as_actor<T>::convert(_2));
} // end do_assign()

} // end functional
} // end detail
} // end thrust

