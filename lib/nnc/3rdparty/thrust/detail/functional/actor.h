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

// Portions of this code are derived from
//
// Manjunath Kudlur's Carbon library
//
// and
//
// Based on Boost.Phoenix v1.2
// Copyright (c) 2001-2002 Joel de Guzman

#pragma once

#include "3rdparty/thrust/detail/config.h"
#include "3rdparty/thrust/tuple.h"
#include "3rdparty/thrust/detail/functional/value.h"
#include "3rdparty/thrust/detail/functional/composite.h"
#include "3rdparty/thrust/detail/functional/operators/assignment_operator.h"
#include "3rdparty/thrust/detail/type_traits/result_of_adaptable_function.h"

namespace thrust
{
namespace detail
{
namespace functional
{

template<typename Action, typename Env>
  struct apply_actor
{
  typedef typename Action::template result<Env>::type type;
};

template<typename Eval>
  struct actor
    : Eval
{
  typedef Eval eval_type;

  __host__ __device__
  THRUST_CONSTEXPR actor();

  __host__ __device__
  actor(const Eval &base);

  __host__ __device__
  typename apply_actor<eval_type, thrust::null_type >::type
  operator()(void) const;

  template<typename T0>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&> >::type
  operator()(T0 &_0) const;

  template<typename T0, typename T1>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&,T1&> >::type
  operator()(T0 &_0, T1 &_1) const;

  template<typename T0, typename T1, typename T2>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&,T1&,T2&> >::type
  operator()(T0 &_0, T1 &_1, T2 &_2) const;

  template<typename T0, typename T1, typename T2, typename T3>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&,T1&,T2&,T3&> >::type
  operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3) const;

  template<typename T0, typename T1, typename T2, typename T3, typename T4>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&,T1&,T2&,T3&,T4&> >::type
  operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4) const;

  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&> >::type
  operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5) const;

  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&> >::type
  operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6) const;

  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&> >::type
  operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6, T7 &_7) const;

  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&> >::type
  operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6, T7 &_7, T8 &_8) const;

  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
  __host__ __device__
  typename apply_actor<eval_type, thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&,T9&> >::type
  operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6, T7 &_7, T8 &_8, T9 &_9) const;

  template<typename T>
  __host__ __device__
  typename assign_result<Eval,T>::type
  operator=(const T &_1) const;
}; // end actor

// in general, as_actor should turn things into values
template<typename T>
  struct as_actor
{
  typedef value<T> type;

  static inline __host__ __device__ type convert(const T &x)
  {
    return val(x);
  } // end convert()
}; // end as_actor

// specialization for things which are already actors
template<typename Eval>
  struct as_actor<actor<Eval> >
{
  typedef actor<Eval> type;

  static inline __host__ __device__ const type &convert(const actor<Eval> &x)
  {
    return x;
  } // end convert()
}; // end as_actor

template<typename T>
  typename as_actor<T>::type
  __host__ __device__
    make_actor(const T &x)
{
  return as_actor<T>::convert(x);
} // end make_actor()

} // end functional

// provide specializations for result_of for nullary, unary, and binary invocations of actor
template<typename Eval>
  struct result_of_adaptable_function<
    thrust::detail::functional::actor<Eval>()
  >
{
  typedef typename thrust::detail::functional::apply_actor<
    thrust::detail::functional::actor<Eval>,
    thrust::null_type
  >::type type;
}; // end result_of

template<typename Eval, typename Arg1>
  struct result_of_adaptable_function<
    thrust::detail::functional::actor<Eval>(Arg1)
  >
{
  typedef typename thrust::detail::functional::apply_actor<
    thrust::detail::functional::actor<Eval>,
    thrust::tuple<Arg1>
  >::type type;
}; // end result_of

template<typename Eval, typename Arg1, typename Arg2>
  struct result_of_adaptable_function<
    thrust::detail::functional::actor<Eval>(Arg1,Arg2)
  >
{
  typedef typename thrust::detail::functional::apply_actor<
    thrust::detail::functional::actor<Eval>,
    thrust::tuple<Arg1,Arg2>
  >::type type;
}; // end result_of

} // end detail
} // end thrust

#include "3rdparty/thrust/detail/functional/actor.inl"

