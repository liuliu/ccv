/*
 *  Copyright 2018 NVIDIA Corporation
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
#include "3rdparty/thrust/detail/type_traits.h"

namespace thrust
{

#if THRUST_CPP_DIALECT >= 2020

using std::remove_cvref;
using std::remove_cvref_t;

#else // Older than C++20.

template <typename T>
struct remove_cvref
{
  typedef typename detail::remove_cv<
    typename detail::remove_reference<T>::type
  >::type type;
};

#if THRUST_CPP_DIALECT >= 2011
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;
#endif

#endif // THRUST_CPP_DIALECT >= 2020

} // end namespace thrust

