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

#include "3rdparty/thrust/detail/config.h"
#include "3rdparty/thrust/detail/functional/composite.h"
#include "3rdparty/thrust/detail/functional/operators/assignment_operator.h"
#include "3rdparty/thrust/functional.h"

namespace thrust
{

namespace detail
{
namespace functional
{

template<typename Eval>
  __host__ __device__
  THRUST_CONSTEXPR actor<Eval>
    ::actor()
      : eval_type()
{}

template<typename Eval>
  __host__ __device__
  actor<Eval>
    ::actor(const Eval &base)
      : eval_type(base)
{}

template<typename Eval>
  __host__ __device__
  typename apply_actor<
    typename actor<Eval>::eval_type,
    typename thrust::null_type
  >::type
    actor<Eval>
      ::operator()(void) const
{
  return eval_type::eval(thrust::null_type());
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0) const
{
  return eval_type::eval(thrust::tie(_0));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0, typename T1>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&,T1&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0, T1 &_1) const
{
  return eval_type::eval(thrust::tie(_0,_1));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0, typename T1, typename T2>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&,T1&,T2&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0, T1 &_1, T2 &_2) const
{
  return eval_type::eval(thrust::tie(_0,_1,_2));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0, typename T1, typename T2, typename T3>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&,T1&,T2&,T3&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3) const
{
  return eval_type::eval(thrust::tie(_0,_1,_2,_3));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0, typename T1, typename T2, typename T3, typename T4>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&,T1&,T2&,T3&,T4&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4) const
{
  return eval_type::eval(thrust::tie(_0,_1,_2,_3,_4));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5) const
{
  return eval_type::eval(thrust::tie(_0,_1,_2,_3,_4,_5));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6) const
{
  return eval_type::eval(thrust::tie(_0,_1,_2,_3,_4,_5,_6));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6, T7 &_7) const
{
  return eval_type::eval(thrust::tie(_0,_1,_2,_3,_4,_5,_6,_7));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6, T7 &_7, T8 &_8) const
{
  return eval_type::eval(thrust::tie(_0,_1,_2,_3,_4,_5,_6,_7,_8));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
    __host__ __device__
    typename apply_actor<
      typename actor<Eval>::eval_type,
      typename thrust::tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&,T9&>
    >::type
      actor<Eval>
        ::operator()(T0 &_0, T1 &_1, T2 &_2, T3 &_3, T4 &_4, T5 &_5, T6 &_6, T7 &_7, T8 &_8, T9 &_9) const
{
  return eval_type::eval(thrust::tie(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9));
} // end basic_environment::operator()

template<typename Eval>
  template<typename T>
    __host__ __device__
    typename assign_result<Eval,T>::type
      actor<Eval>
        ::operator=(const T& _1) const
{
  return do_assign(*this,_1);
} // end actor::operator=()

} // end functional
} // end detail
} // end thrust
