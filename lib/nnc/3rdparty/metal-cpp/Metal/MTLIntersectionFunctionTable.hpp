//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLIntersectionFunctionTable.hpp
//
// Copyright 2020-2023 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

#include <Foundation/Foundation.hpp>

#include "MTLIntersectionFunctionTable.hpp"
#include "MTLResource.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
_MTL_OPTIONS(NS::UInteger, IntersectionFunctionSignature) {
    IntersectionFunctionSignatureNone = 0,
    IntersectionFunctionSignatureInstancing = 1,
    IntersectionFunctionSignatureTriangleData = 2,
    IntersectionFunctionSignatureWorldSpaceData = 4,
    IntersectionFunctionSignatureInstanceMotion = 8,
    IntersectionFunctionSignaturePrimitiveMotion = 16,
    IntersectionFunctionSignatureExtendedLimits = 32,
    IntersectionFunctionSignatureMaxLevels = 64,
    IntersectionFunctionSignatureCurveData = 128,
};

class IntersectionFunctionTableDescriptor : public NS::Copying<IntersectionFunctionTableDescriptor>
{
public:
    static class IntersectionFunctionTableDescriptor* alloc();

    class IntersectionFunctionTableDescriptor*        init();

    static class IntersectionFunctionTableDescriptor* intersectionFunctionTableDescriptor();

    NS::UInteger                                      functionCount() const;
    void                                              setFunctionCount(NS::UInteger functionCount);
};

class IntersectionFunctionTable : public NS::Referencing<IntersectionFunctionTable, Resource>
{
public:
    void            setBuffer(const class Buffer* buffer, NS::UInteger offset, NS::UInteger index);

    void            setBuffers(const class Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range);

    MTL::ResourceID gpuResourceID() const;

    void            setFunction(const class FunctionHandle* function, NS::UInteger index);

    void            setFunctions(const class FunctionHandle* const functions[], NS::Range range);

    void            setOpaqueTriangleIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::UInteger index);

    void            setOpaqueTriangleIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::Range range);

    void            setOpaqueCurveIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::UInteger index);

    void            setOpaqueCurveIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::Range range);

    void            setVisibleFunctionTable(const class VisibleFunctionTable* functionTable, NS::UInteger bufferIndex);

    void            setVisibleFunctionTables(const class VisibleFunctionTable* const functionTables[], NS::Range bufferRange);
};

}

// static method: alloc
_MTL_INLINE MTL::IntersectionFunctionTableDescriptor* MTL::IntersectionFunctionTableDescriptor::alloc()
{
    return NS::Object::alloc<MTL::IntersectionFunctionTableDescriptor>(_MTL_PRIVATE_CLS(MTLIntersectionFunctionTableDescriptor));
}

// method: init
_MTL_INLINE MTL::IntersectionFunctionTableDescriptor* MTL::IntersectionFunctionTableDescriptor::init()
{
    return NS::Object::init<MTL::IntersectionFunctionTableDescriptor>();
}

// static method: intersectionFunctionTableDescriptor
_MTL_INLINE MTL::IntersectionFunctionTableDescriptor* MTL::IntersectionFunctionTableDescriptor::intersectionFunctionTableDescriptor()
{
    return Object::sendMessage<MTL::IntersectionFunctionTableDescriptor*>(_MTL_PRIVATE_CLS(MTLIntersectionFunctionTableDescriptor), _MTL_PRIVATE_SEL(intersectionFunctionTableDescriptor));
}

// property: functionCount
_MTL_INLINE NS::UInteger MTL::IntersectionFunctionTableDescriptor::functionCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(functionCount));
}

_MTL_INLINE void MTL::IntersectionFunctionTableDescriptor::setFunctionCount(NS::UInteger functionCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunctionCount_), functionCount);
}

// method: setBuffer:offset:atIndex:
_MTL_INLINE void MTL::IntersectionFunctionTable::setBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffer_offset_atIndex_), buffer, offset, index);
}

// method: setBuffers:offsets:withRange:
_MTL_INLINE void MTL::IntersectionFunctionTable::setBuffers(const MTL::Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffers_offsets_withRange_), buffers, offsets, range);
}

// property: gpuResourceID
_MTL_INLINE MTL::ResourceID MTL::IntersectionFunctionTable::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

// method: setFunction:atIndex:
_MTL_INLINE void MTL::IntersectionFunctionTable::setFunction(const MTL::FunctionHandle* function, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunction_atIndex_), function, index);
}

// method: setFunctions:withRange:
_MTL_INLINE void MTL::IntersectionFunctionTable::setFunctions(const MTL::FunctionHandle* const functions[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunctions_withRange_), functions, range);
}

// method: setOpaqueTriangleIntersectionFunctionWithSignature:atIndex:
_MTL_INLINE void MTL::IntersectionFunctionTable::setOpaqueTriangleIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOpaqueTriangleIntersectionFunctionWithSignature_atIndex_), signature, index);
}

// method: setOpaqueTriangleIntersectionFunctionWithSignature:withRange:
_MTL_INLINE void MTL::IntersectionFunctionTable::setOpaqueTriangleIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOpaqueTriangleIntersectionFunctionWithSignature_withRange_), signature, range);
}

// method: setOpaqueCurveIntersectionFunctionWithSignature:atIndex:
_MTL_INLINE void MTL::IntersectionFunctionTable::setOpaqueCurveIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOpaqueCurveIntersectionFunctionWithSignature_atIndex_), signature, index);
}

// method: setOpaqueCurveIntersectionFunctionWithSignature:withRange:
_MTL_INLINE void MTL::IntersectionFunctionTable::setOpaqueCurveIntersectionFunction(MTL::IntersectionFunctionSignature signature, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOpaqueCurveIntersectionFunctionWithSignature_withRange_), signature, range);
}

// method: setVisibleFunctionTable:atBufferIndex:
_MTL_INLINE void MTL::IntersectionFunctionTable::setVisibleFunctionTable(const MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibleFunctionTable_atBufferIndex_), functionTable, bufferIndex);
}

// method: setVisibleFunctionTables:withBufferRange:
_MTL_INLINE void MTL::IntersectionFunctionTable::setVisibleFunctionTables(const MTL::VisibleFunctionTable* const functionTables[], NS::Range bufferRange)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibleFunctionTables_withBufferRange_), functionTables, bufferRange);
}
