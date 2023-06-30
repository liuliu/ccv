//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLAccelerationStructure.hpp
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

#include "MTLAccelerationStructure.hpp"
#include "MTLAccelerationStructureTypes.hpp"
#include "MTLResource.hpp"
#include "MTLStageInputOutputDescriptor.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
_MTL_OPTIONS(NS::UInteger, AccelerationStructureUsage) {
    AccelerationStructureUsageNone = 0,
    AccelerationStructureUsageRefit = 1,
    AccelerationStructureUsagePreferFastBuild = 2,
    AccelerationStructureUsageExtendedLimits = 4,
};

_MTL_OPTIONS(uint32_t, AccelerationStructureInstanceOptions) {
    AccelerationStructureInstanceOptionNone = 0,
    AccelerationStructureInstanceOptionDisableTriangleCulling = 1,
    AccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise = 2,
    AccelerationStructureInstanceOptionOpaque = 4,
    AccelerationStructureInstanceOptionNonOpaque = 8,
};

class AccelerationStructureDescriptor : public NS::Copying<AccelerationStructureDescriptor>
{
public:
    static class AccelerationStructureDescriptor* alloc();

    class AccelerationStructureDescriptor*        init();

    MTL::AccelerationStructureUsage               usage() const;
    void                                          setUsage(MTL::AccelerationStructureUsage usage);
};

class AccelerationStructureGeometryDescriptor : public NS::Copying<AccelerationStructureGeometryDescriptor>
{
public:
    static class AccelerationStructureGeometryDescriptor* alloc();

    class AccelerationStructureGeometryDescriptor*        init();

    NS::UInteger                                          intersectionFunctionTableOffset() const;
    void                                                  setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset);

    bool                                                  opaque() const;
    void                                                  setOpaque(bool opaque);

    bool                                                  allowDuplicateIntersectionFunctionInvocation() const;
    void                                                  setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation);

    NS::String*                                           label() const;
    void                                                  setLabel(const NS::String* label);

    class Buffer*                                         primitiveDataBuffer() const;
    void                                                  setPrimitiveDataBuffer(const class Buffer* primitiveDataBuffer);

    NS::UInteger                                          primitiveDataBufferOffset() const;
    void                                                  setPrimitiveDataBufferOffset(NS::UInteger primitiveDataBufferOffset);

    NS::UInteger                                          primitiveDataStride() const;
    void                                                  setPrimitiveDataStride(NS::UInteger primitiveDataStride);

    NS::UInteger                                          primitiveDataElementSize() const;
    void                                                  setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize);
};

_MTL_ENUM(uint32_t, MotionBorderMode) {
    MotionBorderModeClamp = 0,
    MotionBorderModeVanish = 1,
};

class PrimitiveAccelerationStructureDescriptor : public NS::Copying<PrimitiveAccelerationStructureDescriptor, MTL::AccelerationStructureDescriptor>
{
public:
    static class PrimitiveAccelerationStructureDescriptor* alloc();

    class PrimitiveAccelerationStructureDescriptor*        init();

    NS::Array*                                             geometryDescriptors() const;
    void                                                   setGeometryDescriptors(const NS::Array* geometryDescriptors);

    MTL::MotionBorderMode                                  motionStartBorderMode() const;
    void                                                   setMotionStartBorderMode(MTL::MotionBorderMode motionStartBorderMode);

    MTL::MotionBorderMode                                  motionEndBorderMode() const;
    void                                                   setMotionEndBorderMode(MTL::MotionBorderMode motionEndBorderMode);

    float                                                  motionStartTime() const;
    void                                                   setMotionStartTime(float motionStartTime);

    float                                                  motionEndTime() const;
    void                                                   setMotionEndTime(float motionEndTime);

    NS::UInteger                                           motionKeyframeCount() const;
    void                                                   setMotionKeyframeCount(NS::UInteger motionKeyframeCount);

    static MTL::PrimitiveAccelerationStructureDescriptor*  descriptor();
};

class AccelerationStructureTriangleGeometryDescriptor : public NS::Copying<AccelerationStructureTriangleGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static class AccelerationStructureTriangleGeometryDescriptor* alloc();

    class AccelerationStructureTriangleGeometryDescriptor*        init();

    class Buffer*                                                 vertexBuffer() const;
    void                                                          setVertexBuffer(const class Buffer* vertexBuffer);

    NS::UInteger                                                  vertexBufferOffset() const;
    void                                                          setVertexBufferOffset(NS::UInteger vertexBufferOffset);

    MTL::AttributeFormat                                          vertexFormat() const;
    void                                                          setVertexFormat(MTL::AttributeFormat vertexFormat);

    NS::UInteger                                                  vertexStride() const;
    void                                                          setVertexStride(NS::UInteger vertexStride);

    class Buffer*                                                 indexBuffer() const;
    void                                                          setIndexBuffer(const class Buffer* indexBuffer);

    NS::UInteger                                                  indexBufferOffset() const;
    void                                                          setIndexBufferOffset(NS::UInteger indexBufferOffset);

    MTL::IndexType                                                indexType() const;
    void                                                          setIndexType(MTL::IndexType indexType);

    NS::UInteger                                                  triangleCount() const;
    void                                                          setTriangleCount(NS::UInteger triangleCount);

    class Buffer*                                                 transformationMatrixBuffer() const;
    void                                                          setTransformationMatrixBuffer(const class Buffer* transformationMatrixBuffer);

    NS::UInteger                                                  transformationMatrixBufferOffset() const;
    void                                                          setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset);

    static MTL::AccelerationStructureTriangleGeometryDescriptor*  descriptor();
};

class AccelerationStructureBoundingBoxGeometryDescriptor : public NS::Copying<AccelerationStructureBoundingBoxGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static class AccelerationStructureBoundingBoxGeometryDescriptor* alloc();

    class AccelerationStructureBoundingBoxGeometryDescriptor*        init();

    class Buffer*                                                    boundingBoxBuffer() const;
    void                                                             setBoundingBoxBuffer(const class Buffer* boundingBoxBuffer);

    NS::UInteger                                                     boundingBoxBufferOffset() const;
    void                                                             setBoundingBoxBufferOffset(NS::UInteger boundingBoxBufferOffset);

    NS::UInteger                                                     boundingBoxStride() const;
    void                                                             setBoundingBoxStride(NS::UInteger boundingBoxStride);

    NS::UInteger                                                     boundingBoxCount() const;
    void                                                             setBoundingBoxCount(NS::UInteger boundingBoxCount);

    static MTL::AccelerationStructureBoundingBoxGeometryDescriptor*  descriptor();
};

class MotionKeyframeData : public NS::Referencing<MotionKeyframeData>
{
public:
    static class MotionKeyframeData* alloc();

    class MotionKeyframeData*        init();

    class Buffer*                    buffer() const;
    void                             setBuffer(const class Buffer* buffer);

    NS::UInteger                     offset() const;
    void                             setOffset(NS::UInteger offset);

    static MTL::MotionKeyframeData*  data();
};

class AccelerationStructureMotionTriangleGeometryDescriptor : public NS::Copying<AccelerationStructureMotionTriangleGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static class AccelerationStructureMotionTriangleGeometryDescriptor* alloc();

    class AccelerationStructureMotionTriangleGeometryDescriptor*        init();

    NS::Array*                                                          vertexBuffers() const;
    void                                                                setVertexBuffers(const NS::Array* vertexBuffers);

    MTL::AttributeFormat                                                vertexFormat() const;
    void                                                                setVertexFormat(MTL::AttributeFormat vertexFormat);

    NS::UInteger                                                        vertexStride() const;
    void                                                                setVertexStride(NS::UInteger vertexStride);

    class Buffer*                                                       indexBuffer() const;
    void                                                                setIndexBuffer(const class Buffer* indexBuffer);

    NS::UInteger                                                        indexBufferOffset() const;
    void                                                                setIndexBufferOffset(NS::UInteger indexBufferOffset);

    MTL::IndexType                                                      indexType() const;
    void                                                                setIndexType(MTL::IndexType indexType);

    NS::UInteger                                                        triangleCount() const;
    void                                                                setTriangleCount(NS::UInteger triangleCount);

    class Buffer*                                                       transformationMatrixBuffer() const;
    void                                                                setTransformationMatrixBuffer(const class Buffer* transformationMatrixBuffer);

    NS::UInteger                                                        transformationMatrixBufferOffset() const;
    void                                                                setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset);

    static MTL::AccelerationStructureMotionTriangleGeometryDescriptor*  descriptor();
};

class AccelerationStructureMotionBoundingBoxGeometryDescriptor : public NS::Copying<AccelerationStructureMotionBoundingBoxGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static class AccelerationStructureMotionBoundingBoxGeometryDescriptor* alloc();

    class AccelerationStructureMotionBoundingBoxGeometryDescriptor*        init();

    NS::Array*                                                             boundingBoxBuffers() const;
    void                                                                   setBoundingBoxBuffers(const NS::Array* boundingBoxBuffers);

    NS::UInteger                                                           boundingBoxStride() const;
    void                                                                   setBoundingBoxStride(NS::UInteger boundingBoxStride);

    NS::UInteger                                                           boundingBoxCount() const;
    void                                                                   setBoundingBoxCount(NS::UInteger boundingBoxCount);

    static MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor*  descriptor();
};

_MTL_ENUM(NS::Integer, CurveType) {
    CurveTypeRound = 0,
    CurveTypeFlat = 1,
};

_MTL_ENUM(NS::Integer, CurveBasis) {
    CurveBasisBSpline = 0,
    CurveBasisCatmullRom = 1,
    CurveBasisLinear = 2,
    CurveBasisBezier = 3,
};

_MTL_ENUM(NS::Integer, CurveEndCaps) {
    CurveEndCapsNone = 0,
    CurveEndCapsDisk = 1,
    CurveEndCapsSphere = 2,
};

class AccelerationStructureCurveGeometryDescriptor : public NS::Copying<AccelerationStructureCurveGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static class AccelerationStructureCurveGeometryDescriptor* alloc();

    class AccelerationStructureCurveGeometryDescriptor*        init();

    class Buffer*                                              controlPointBuffer() const;
    void                                                       setControlPointBuffer(const class Buffer* controlPointBuffer);

    NS::UInteger                                               controlPointBufferOffset() const;
    void                                                       setControlPointBufferOffset(NS::UInteger controlPointBufferOffset);

    NS::UInteger                                               controlPointCount() const;
    void                                                       setControlPointCount(NS::UInteger controlPointCount);

    NS::UInteger                                               controlPointStride() const;
    void                                                       setControlPointStride(NS::UInteger controlPointStride);

    MTL::AttributeFormat                                       controlPointFormat() const;
    void                                                       setControlPointFormat(MTL::AttributeFormat controlPointFormat);

    class Buffer*                                              radiusBuffer() const;
    void                                                       setRadiusBuffer(const class Buffer* radiusBuffer);

    NS::UInteger                                               radiusBufferOffset() const;
    void                                                       setRadiusBufferOffset(NS::UInteger radiusBufferOffset);

    MTL::AttributeFormat                                       radiusFormat() const;
    void                                                       setRadiusFormat(MTL::AttributeFormat radiusFormat);

    NS::UInteger                                               radiusStride() const;
    void                                                       setRadiusStride(NS::UInteger radiusStride);

    class Buffer*                                              indexBuffer() const;
    void                                                       setIndexBuffer(const class Buffer* indexBuffer);

    NS::UInteger                                               indexBufferOffset() const;
    void                                                       setIndexBufferOffset(NS::UInteger indexBufferOffset);

    MTL::IndexType                                             indexType() const;
    void                                                       setIndexType(MTL::IndexType indexType);

    NS::UInteger                                               segmentCount() const;
    void                                                       setSegmentCount(NS::UInteger segmentCount);

    NS::UInteger                                               segmentControlPointCount() const;
    void                                                       setSegmentControlPointCount(NS::UInteger segmentControlPointCount);

    MTL::CurveType                                             curveType() const;
    void                                                       setCurveType(MTL::CurveType curveType);

    MTL::CurveBasis                                            curveBasis() const;
    void                                                       setCurveBasis(MTL::CurveBasis curveBasis);

    MTL::CurveEndCaps                                          curveEndCaps() const;
    void                                                       setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);

    static MTL::AccelerationStructureCurveGeometryDescriptor*  descriptor();
};

class AccelerationStructureMotionCurveGeometryDescriptor : public NS::Copying<AccelerationStructureMotionCurveGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static class AccelerationStructureMotionCurveGeometryDescriptor* alloc();

    class AccelerationStructureMotionCurveGeometryDescriptor*        init();

    NS::Array*                                                       controlPointBuffers() const;
    void                                                             setControlPointBuffers(const NS::Array* controlPointBuffers);

    NS::UInteger                                                     controlPointCount() const;
    void                                                             setControlPointCount(NS::UInteger controlPointCount);

    NS::UInteger                                                     controlPointStride() const;
    void                                                             setControlPointStride(NS::UInteger controlPointStride);

    MTL::AttributeFormat                                             controlPointFormat() const;
    void                                                             setControlPointFormat(MTL::AttributeFormat controlPointFormat);

    NS::Array*                                                       radiusBuffers() const;
    void                                                             setRadiusBuffers(const NS::Array* radiusBuffers);

    MTL::AttributeFormat                                             radiusFormat() const;
    void                                                             setRadiusFormat(MTL::AttributeFormat radiusFormat);

    NS::UInteger                                                     radiusStride() const;
    void                                                             setRadiusStride(NS::UInteger radiusStride);

    class Buffer*                                                    indexBuffer() const;
    void                                                             setIndexBuffer(const class Buffer* indexBuffer);

    NS::UInteger                                                     indexBufferOffset() const;
    void                                                             setIndexBufferOffset(NS::UInteger indexBufferOffset);

    MTL::IndexType                                                   indexType() const;
    void                                                             setIndexType(MTL::IndexType indexType);

    NS::UInteger                                                     segmentCount() const;
    void                                                             setSegmentCount(NS::UInteger segmentCount);

    NS::UInteger                                                     segmentControlPointCount() const;
    void                                                             setSegmentControlPointCount(NS::UInteger segmentControlPointCount);

    MTL::CurveType                                                   curveType() const;
    void                                                             setCurveType(MTL::CurveType curveType);

    MTL::CurveBasis                                                  curveBasis() const;
    void                                                             setCurveBasis(MTL::CurveBasis curveBasis);

    MTL::CurveEndCaps                                                curveEndCaps() const;
    void                                                             setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);

    static MTL::AccelerationStructureMotionCurveGeometryDescriptor*  descriptor();
};

struct AccelerationStructureInstanceDescriptor
{
    MTL::PackedFloat4x3                       transformationMatrix;
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  accelerationStructureIndex;
} _MTL_PACKED;

struct AccelerationStructureUserIDInstanceDescriptor
{
    MTL::PackedFloat4x3                       transformationMatrix;
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  accelerationStructureIndex;
    uint32_t                                  userID;
} _MTL_PACKED;

_MTL_ENUM(NS::UInteger, AccelerationStructureInstanceDescriptorType) {
    AccelerationStructureInstanceDescriptorTypeDefault = 0,
    AccelerationStructureInstanceDescriptorTypeUserID = 1,
    AccelerationStructureInstanceDescriptorTypeMotion = 2,
    AccelerationStructureInstanceDescriptorTypeIndirect = 3,
    AccelerationStructureInstanceDescriptorTypeIndirectMotion = 4,
};

struct AccelerationStructureMotionInstanceDescriptor
{
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  accelerationStructureIndex;
    uint32_t                                  userID;
    uint32_t                                  motionTransformsStartIndex;
    uint32_t                                  motionTransformsCount;
    MTL::MotionBorderMode                     motionStartBorderMode;
    MTL::MotionBorderMode                     motionEndBorderMode;
    float                                     motionStartTime;
    float                                     motionEndTime;
} _MTL_PACKED;

struct IndirectAccelerationStructureInstanceDescriptor
{
    MTL::PackedFloat4x3                       transformationMatrix;
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  userID;
    MTL::ResourceID                           accelerationStructureID;
} _MTL_PACKED;

struct IndirectAccelerationStructureMotionInstanceDescriptor
{
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  userID;
    MTL::ResourceID                           accelerationStructureID;
    uint32_t                                  motionTransformsStartIndex;
    uint32_t                                  motionTransformsCount;
    MTL::MotionBorderMode                     motionStartBorderMode;
    MTL::MotionBorderMode                     motionEndBorderMode;
    float                                     motionStartTime;
    float                                     motionEndTime;
} _MTL_PACKED;

class InstanceAccelerationStructureDescriptor : public NS::Copying<InstanceAccelerationStructureDescriptor, MTL::AccelerationStructureDescriptor>
{
public:
    static class InstanceAccelerationStructureDescriptor* alloc();

    class InstanceAccelerationStructureDescriptor*        init();

    class Buffer*                                         instanceDescriptorBuffer() const;
    void                                                  setInstanceDescriptorBuffer(const class Buffer* instanceDescriptorBuffer);

    NS::UInteger                                          instanceDescriptorBufferOffset() const;
    void                                                  setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset);

    NS::UInteger                                          instanceDescriptorStride() const;
    void                                                  setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);

    NS::UInteger                                          instanceCount() const;
    void                                                  setInstanceCount(NS::UInteger instanceCount);

    NS::Array*                                            instancedAccelerationStructures() const;
    void                                                  setInstancedAccelerationStructures(const NS::Array* instancedAccelerationStructures);

    MTL::AccelerationStructureInstanceDescriptorType      instanceDescriptorType() const;
    void                                                  setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);

    class Buffer*                                         motionTransformBuffer() const;
    void                                                  setMotionTransformBuffer(const class Buffer* motionTransformBuffer);

    NS::UInteger                                          motionTransformBufferOffset() const;
    void                                                  setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset);

    NS::UInteger                                          motionTransformCount() const;
    void                                                  setMotionTransformCount(NS::UInteger motionTransformCount);

    static MTL::InstanceAccelerationStructureDescriptor*  descriptor();
};

class IndirectInstanceAccelerationStructureDescriptor : public NS::Copying<IndirectInstanceAccelerationStructureDescriptor, MTL::AccelerationStructureDescriptor>
{
public:
    static class IndirectInstanceAccelerationStructureDescriptor* alloc();

    class IndirectInstanceAccelerationStructureDescriptor*        init();

    class Buffer*                                                 instanceDescriptorBuffer() const;
    void                                                          setInstanceDescriptorBuffer(const class Buffer* instanceDescriptorBuffer);

    NS::UInteger                                                  instanceDescriptorBufferOffset() const;
    void                                                          setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset);

    NS::UInteger                                                  instanceDescriptorStride() const;
    void                                                          setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);

    NS::UInteger                                                  maxInstanceCount() const;
    void                                                          setMaxInstanceCount(NS::UInteger maxInstanceCount);

    class Buffer*                                                 instanceCountBuffer() const;
    void                                                          setInstanceCountBuffer(const class Buffer* instanceCountBuffer);

    NS::UInteger                                                  instanceCountBufferOffset() const;
    void                                                          setInstanceCountBufferOffset(NS::UInteger instanceCountBufferOffset);

    MTL::AccelerationStructureInstanceDescriptorType              instanceDescriptorType() const;
    void                                                          setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);

    class Buffer*                                                 motionTransformBuffer() const;
    void                                                          setMotionTransformBuffer(const class Buffer* motionTransformBuffer);

    NS::UInteger                                                  motionTransformBufferOffset() const;
    void                                                          setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset);

    NS::UInteger                                                  maxMotionTransformCount() const;
    void                                                          setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount);

    class Buffer*                                                 motionTransformCountBuffer() const;
    void                                                          setMotionTransformCountBuffer(const class Buffer* motionTransformCountBuffer);

    NS::UInteger                                                  motionTransformCountBufferOffset() const;
    void                                                          setMotionTransformCountBufferOffset(NS::UInteger motionTransformCountBufferOffset);

    static MTL::IndirectInstanceAccelerationStructureDescriptor*  descriptor();
};

class AccelerationStructure : public NS::Referencing<AccelerationStructure, Resource>
{
public:
    NS::UInteger    size() const;

    MTL::ResourceID gpuResourceID() const;
};

}

// static method: alloc
_MTL_INLINE MTL::AccelerationStructureDescriptor* MTL::AccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureDescriptor));
}

// method: init
_MTL_INLINE MTL::AccelerationStructureDescriptor* MTL::AccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureDescriptor>();
}

// property: usage
_MTL_INLINE MTL::AccelerationStructureUsage MTL::AccelerationStructureDescriptor::usage() const
{
    return Object::sendMessage<MTL::AccelerationStructureUsage>(this, _MTL_PRIVATE_SEL(usage));
}

_MTL_INLINE void MTL::AccelerationStructureDescriptor::setUsage(MTL::AccelerationStructureUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setUsage_), usage);
}

// static method: alloc
_MTL_INLINE MTL::AccelerationStructureGeometryDescriptor* MTL::AccelerationStructureGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureGeometryDescriptor));
}

// method: init
_MTL_INLINE MTL::AccelerationStructureGeometryDescriptor* MTL::AccelerationStructureGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureGeometryDescriptor>();
}

// property: intersectionFunctionTableOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::intersectionFunctionTableOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(intersectionFunctionTableOffset));
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIntersectionFunctionTableOffset_), intersectionFunctionTableOffset);
}

// property: opaque
_MTL_INLINE bool MTL::AccelerationStructureGeometryDescriptor::opaque() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(opaque));
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setOpaque(bool opaque)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOpaque_), opaque);
}

// property: allowDuplicateIntersectionFunctionInvocation
_MTL_INLINE bool MTL::AccelerationStructureGeometryDescriptor::allowDuplicateIntersectionFunctionInvocation() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(allowDuplicateIntersectionFunctionInvocation));
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAllowDuplicateIntersectionFunctionInvocation_), allowDuplicateIntersectionFunctionInvocation);
}

// property: label
_MTL_INLINE NS::String* MTL::AccelerationStructureGeometryDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

// property: primitiveDataBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureGeometryDescriptor::primitiveDataBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(primitiveDataBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataBuffer(const MTL::Buffer* primitiveDataBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataBuffer_), primitiveDataBuffer);
}

// property: primitiveDataBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::primitiveDataBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(primitiveDataBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataBufferOffset(NS::UInteger primitiveDataBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataBufferOffset_), primitiveDataBufferOffset);
}

// property: primitiveDataStride
_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::primitiveDataStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(primitiveDataStride));
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataStride(NS::UInteger primitiveDataStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataStride_), primitiveDataStride);
}

// property: primitiveDataElementSize
_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::primitiveDataElementSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(primitiveDataElementSize));
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataElementSize_), primitiveDataElementSize);
}

// static method: alloc
_MTL_INLINE MTL::PrimitiveAccelerationStructureDescriptor* MTL::PrimitiveAccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::PrimitiveAccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTLPrimitiveAccelerationStructureDescriptor));
}

// method: init
_MTL_INLINE MTL::PrimitiveAccelerationStructureDescriptor* MTL::PrimitiveAccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL::PrimitiveAccelerationStructureDescriptor>();
}

// property: geometryDescriptors
_MTL_INLINE NS::Array* MTL::PrimitiveAccelerationStructureDescriptor::geometryDescriptors() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(geometryDescriptors));
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setGeometryDescriptors(const NS::Array* geometryDescriptors)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setGeometryDescriptors_), geometryDescriptors);
}

// property: motionStartBorderMode
_MTL_INLINE MTL::MotionBorderMode MTL::PrimitiveAccelerationStructureDescriptor::motionStartBorderMode() const
{
    return Object::sendMessage<MTL::MotionBorderMode>(this, _MTL_PRIVATE_SEL(motionStartBorderMode));
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionStartBorderMode(MTL::MotionBorderMode motionStartBorderMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionStartBorderMode_), motionStartBorderMode);
}

// property: motionEndBorderMode
_MTL_INLINE MTL::MotionBorderMode MTL::PrimitiveAccelerationStructureDescriptor::motionEndBorderMode() const
{
    return Object::sendMessage<MTL::MotionBorderMode>(this, _MTL_PRIVATE_SEL(motionEndBorderMode));
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionEndBorderMode(MTL::MotionBorderMode motionEndBorderMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionEndBorderMode_), motionEndBorderMode);
}

// property: motionStartTime
_MTL_INLINE float MTL::PrimitiveAccelerationStructureDescriptor::motionStartTime() const
{
    return Object::sendMessage<float>(this, _MTL_PRIVATE_SEL(motionStartTime));
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionStartTime(float motionStartTime)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionStartTime_), motionStartTime);
}

// property: motionEndTime
_MTL_INLINE float MTL::PrimitiveAccelerationStructureDescriptor::motionEndTime() const
{
    return Object::sendMessage<float>(this, _MTL_PRIVATE_SEL(motionEndTime));
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionEndTime(float motionEndTime)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionEndTime_), motionEndTime);
}

// property: motionKeyframeCount
_MTL_INLINE NS::UInteger MTL::PrimitiveAccelerationStructureDescriptor::motionKeyframeCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionKeyframeCount));
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionKeyframeCount(NS::UInteger motionKeyframeCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionKeyframeCount_), motionKeyframeCount);
}

// static method: descriptor
_MTL_INLINE MTL::PrimitiveAccelerationStructureDescriptor* MTL::PrimitiveAccelerationStructureDescriptor::descriptor()
{
    return Object::sendMessage<MTL::PrimitiveAccelerationStructureDescriptor*>(_MTL_PRIVATE_CLS(MTLPrimitiveAccelerationStructureDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

// static method: alloc
_MTL_INLINE MTL::AccelerationStructureTriangleGeometryDescriptor* MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureTriangleGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureTriangleGeometryDescriptor));
}

// method: init
_MTL_INLINE MTL::AccelerationStructureTriangleGeometryDescriptor* MTL::AccelerationStructureTriangleGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureTriangleGeometryDescriptor>();
}

// property: vertexBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureTriangleGeometryDescriptor::vertexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(vertexBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexBuffer(const MTL::Buffer* vertexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffer_), vertexBuffer);
}

// property: vertexBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::vertexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(vertexBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexBufferOffset(NS::UInteger vertexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBufferOffset_), vertexBufferOffset);
}

// property: vertexFormat
_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureTriangleGeometryDescriptor::vertexFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(vertexFormat));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexFormat_), vertexFormat);
}

// property: vertexStride
_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::vertexStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(vertexStride));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexStride_), vertexStride);
}

// property: indexBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureTriangleGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setIndexBuffer(const MTL::Buffer* indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

// property: indexBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::indexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(indexBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBufferOffset_), indexBufferOffset);
}

// property: indexType
_MTL_INLINE MTL::IndexType MTL::AccelerationStructureTriangleGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

// property: triangleCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::triangleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(triangleCount));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTriangleCount_), triangleCount);
}

// property: transformationMatrixBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(transformationMatrixBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixBuffer(const MTL::Buffer* transformationMatrixBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBuffer_), transformationMatrixBuffer);
}

// property: transformationMatrixBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(transformationMatrixBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBufferOffset_), transformationMatrixBufferOffset);
}

// static method: descriptor
_MTL_INLINE MTL::AccelerationStructureTriangleGeometryDescriptor* MTL::AccelerationStructureTriangleGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureTriangleGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureTriangleGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

// static method: alloc
_MTL_INLINE MTL::AccelerationStructureBoundingBoxGeometryDescriptor* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureBoundingBoxGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureBoundingBoxGeometryDescriptor));
}

// method: init
_MTL_INLINE MTL::AccelerationStructureBoundingBoxGeometryDescriptor* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureBoundingBoxGeometryDescriptor>();
}

// property: boundingBoxBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(boundingBoxBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxBuffer(const MTL::Buffer* boundingBoxBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxBuffer_), boundingBoxBuffer);
}

// property: boundingBoxBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxBufferOffset(NS::UInteger boundingBoxBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxBufferOffset_), boundingBoxBufferOffset);
}

// property: boundingBoxStride
_MTL_INLINE NS::UInteger MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxStride));
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxStride_), boundingBoxStride);
}

// property: boundingBoxCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxCount));
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxCount_), boundingBoxCount);
}

// static method: descriptor
_MTL_INLINE MTL::AccelerationStructureBoundingBoxGeometryDescriptor* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureBoundingBoxGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureBoundingBoxGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

// static method: alloc
_MTL_INLINE MTL::MotionKeyframeData* MTL::MotionKeyframeData::alloc()
{
    return NS::Object::alloc<MTL::MotionKeyframeData>(_MTL_PRIVATE_CLS(MTLMotionKeyframeData));
}

// method: init
_MTL_INLINE MTL::MotionKeyframeData* MTL::MotionKeyframeData::init()
{
    return NS::Object::init<MTL::MotionKeyframeData>();
}

// property: buffer
_MTL_INLINE MTL::Buffer* MTL::MotionKeyframeData::buffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(buffer));
}

_MTL_INLINE void MTL::MotionKeyframeData::setBuffer(const MTL::Buffer* buffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffer_), buffer);
}

// property: offset
_MTL_INLINE NS::UInteger MTL::MotionKeyframeData::offset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(offset));
}

_MTL_INLINE void MTL::MotionKeyframeData::setOffset(NS::UInteger offset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOffset_), offset);
}

// static method: data
_MTL_INLINE MTL::MotionKeyframeData* MTL::MotionKeyframeData::data()
{
    return Object::sendMessage<MTL::MotionKeyframeData*>(_MTL_PRIVATE_CLS(MTLMotionKeyframeData), _MTL_PRIVATE_SEL(data));
}

// static method: alloc
_MTL_INLINE MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureMotionTriangleGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionTriangleGeometryDescriptor));
}

// method: init
_MTL_INLINE MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureMotionTriangleGeometryDescriptor>();
}

// property: vertexBuffers
_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::vertexBuffers() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(vertexBuffers));
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexBuffers(const NS::Array* vertexBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffers_), vertexBuffers);
}

// property: vertexFormat
_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureMotionTriangleGeometryDescriptor::vertexFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(vertexFormat));
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexFormat_), vertexFormat);
}

// property: vertexStride
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::vertexStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(vertexStride));
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexStride_), vertexStride);
}

// property: indexBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexBuffer(const MTL::Buffer* indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

// property: indexBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::indexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(indexBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBufferOffset_), indexBufferOffset);
}

// property: indexType
_MTL_INLINE MTL::IndexType MTL::AccelerationStructureMotionTriangleGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

// property: triangleCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::triangleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(triangleCount));
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTriangleCount_), triangleCount);
}

// property: transformationMatrixBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(transformationMatrixBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixBuffer(const MTL::Buffer* transformationMatrixBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBuffer_), transformationMatrixBuffer);
}

// property: transformationMatrixBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(transformationMatrixBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBufferOffset_), transformationMatrixBufferOffset);
}

// static method: descriptor
_MTL_INLINE MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureMotionTriangleGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionTriangleGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

// static method: alloc
_MTL_INLINE MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionBoundingBoxGeometryDescriptor));
}

// method: init
_MTL_INLINE MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor>();
}

// property: boundingBoxBuffers
_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxBuffers() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(boundingBoxBuffers));
}

_MTL_INLINE void MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxBuffers(const NS::Array* boundingBoxBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxBuffers_), boundingBoxBuffers);
}

// property: boundingBoxStride
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxStride));
}

_MTL_INLINE void MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxStride_), boundingBoxStride);
}

// property: boundingBoxCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxCount));
}

_MTL_INLINE void MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxCount_), boundingBoxCount);
}

// static method: descriptor
_MTL_INLINE MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionBoundingBoxGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

// static method: alloc
_MTL_INLINE MTL::AccelerationStructureCurveGeometryDescriptor* MTL::AccelerationStructureCurveGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureCurveGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureCurveGeometryDescriptor));
}

// method: init
_MTL_INLINE MTL::AccelerationStructureCurveGeometryDescriptor* MTL::AccelerationStructureCurveGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureCurveGeometryDescriptor>();
}

// property: controlPointBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureCurveGeometryDescriptor::controlPointBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(controlPointBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointBuffer(const MTL::Buffer* controlPointBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointBuffer_), controlPointBuffer);
}

// property: controlPointBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::controlPointBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointBufferOffset(NS::UInteger controlPointBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointBufferOffset_), controlPointBufferOffset);
}

// property: controlPointCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::controlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointCount));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointCount_), controlPointCount);
}

// property: controlPointStride
_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::controlPointStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointStride));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointStride_), controlPointStride);
}

// property: controlPointFormat
_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureCurveGeometryDescriptor::controlPointFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(controlPointFormat));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointFormat_), controlPointFormat);
}

// property: radiusBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureCurveGeometryDescriptor::radiusBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(radiusBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusBuffer(const MTL::Buffer* radiusBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusBuffer_), radiusBuffer);
}

// property: radiusBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::radiusBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(radiusBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusBufferOffset(NS::UInteger radiusBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusBufferOffset_), radiusBufferOffset);
}

// property: radiusFormat
_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureCurveGeometryDescriptor::radiusFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(radiusFormat));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusFormat_), radiusFormat);
}

// property: radiusStride
_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::radiusStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(radiusStride));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusStride_), radiusStride);
}

// property: indexBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureCurveGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setIndexBuffer(const MTL::Buffer* indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

// property: indexBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::indexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(indexBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBufferOffset_), indexBufferOffset);
}

// property: indexType
_MTL_INLINE MTL::IndexType MTL::AccelerationStructureCurveGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

// property: segmentCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::segmentCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentCount));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentCount_), segmentCount);
}

// property: segmentControlPointCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::segmentControlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentControlPointCount));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentControlPointCount_), segmentControlPointCount);
}

// property: curveType
_MTL_INLINE MTL::CurveType MTL::AccelerationStructureCurveGeometryDescriptor::curveType() const
{
    return Object::sendMessage<MTL::CurveType>(this, _MTL_PRIVATE_SEL(curveType));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveType_), curveType);
}

// property: curveBasis
_MTL_INLINE MTL::CurveBasis MTL::AccelerationStructureCurveGeometryDescriptor::curveBasis() const
{
    return Object::sendMessage<MTL::CurveBasis>(this, _MTL_PRIVATE_SEL(curveBasis));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveBasis_), curveBasis);
}

// property: curveEndCaps
_MTL_INLINE MTL::CurveEndCaps MTL::AccelerationStructureCurveGeometryDescriptor::curveEndCaps() const
{
    return Object::sendMessage<MTL::CurveEndCaps>(this, _MTL_PRIVATE_SEL(curveEndCaps));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveEndCaps_), curveEndCaps);
}

// static method: descriptor
_MTL_INLINE MTL::AccelerationStructureCurveGeometryDescriptor* MTL::AccelerationStructureCurveGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureCurveGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureCurveGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

// static method: alloc
_MTL_INLINE MTL::AccelerationStructureMotionCurveGeometryDescriptor* MTL::AccelerationStructureMotionCurveGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureMotionCurveGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionCurveGeometryDescriptor));
}

// method: init
_MTL_INLINE MTL::AccelerationStructureMotionCurveGeometryDescriptor* MTL::AccelerationStructureMotionCurveGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureMotionCurveGeometryDescriptor>();
}

// property: controlPointBuffers
_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointBuffers() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(controlPointBuffers));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointBuffers(const NS::Array* controlPointBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointBuffers_), controlPointBuffers);
}

// property: controlPointCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointCount));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointCount_), controlPointCount);
}

// property: controlPointStride
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointStride));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointStride_), controlPointStride);
}

// property: controlPointFormat
_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(controlPointFormat));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointFormat_), controlPointFormat);
}

// property: radiusBuffers
_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionCurveGeometryDescriptor::radiusBuffers() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(radiusBuffers));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusBuffers(const NS::Array* radiusBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusBuffers_), radiusBuffers);
}

// property: radiusFormat
_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureMotionCurveGeometryDescriptor::radiusFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(radiusFormat));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusFormat_), radiusFormat);
}

// property: radiusStride
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::radiusStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(radiusStride));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusStride_), radiusStride);
}

// property: indexBuffer
_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureMotionCurveGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setIndexBuffer(const MTL::Buffer* indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

// property: indexBufferOffset
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::indexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(indexBufferOffset));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBufferOffset_), indexBufferOffset);
}

// property: indexType
_MTL_INLINE MTL::IndexType MTL::AccelerationStructureMotionCurveGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

// property: segmentCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::segmentCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentCount));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentCount_), segmentCount);
}

// property: segmentControlPointCount
_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::segmentControlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentControlPointCount));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentControlPointCount_), segmentControlPointCount);
}

// property: curveType
_MTL_INLINE MTL::CurveType MTL::AccelerationStructureMotionCurveGeometryDescriptor::curveType() const
{
    return Object::sendMessage<MTL::CurveType>(this, _MTL_PRIVATE_SEL(curveType));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveType_), curveType);
}

// property: curveBasis
_MTL_INLINE MTL::CurveBasis MTL::AccelerationStructureMotionCurveGeometryDescriptor::curveBasis() const
{
    return Object::sendMessage<MTL::CurveBasis>(this, _MTL_PRIVATE_SEL(curveBasis));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveBasis_), curveBasis);
}

// property: curveEndCaps
_MTL_INLINE MTL::CurveEndCaps MTL::AccelerationStructureMotionCurveGeometryDescriptor::curveEndCaps() const
{
    return Object::sendMessage<MTL::CurveEndCaps>(this, _MTL_PRIVATE_SEL(curveEndCaps));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveEndCaps_), curveEndCaps);
}

// static method: descriptor
_MTL_INLINE MTL::AccelerationStructureMotionCurveGeometryDescriptor* MTL::AccelerationStructureMotionCurveGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureMotionCurveGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionCurveGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

// static method: alloc
_MTL_INLINE MTL::InstanceAccelerationStructureDescriptor* MTL::InstanceAccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::InstanceAccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTLInstanceAccelerationStructureDescriptor));
}

// method: init
_MTL_INLINE MTL::InstanceAccelerationStructureDescriptor* MTL::InstanceAccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL::InstanceAccelerationStructureDescriptor>();
}

// property: instanceDescriptorBuffer
_MTL_INLINE MTL::Buffer* MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(instanceDescriptorBuffer));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(const MTL::Buffer* instanceDescriptorBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBuffer_), instanceDescriptorBuffer);
}

// property: instanceDescriptorBufferOffset
_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorBufferOffset));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBufferOffset_), instanceDescriptorBufferOffset);
}

// property: instanceDescriptorStride
_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorStride));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorStride_), instanceDescriptorStride);
}

// property: instanceCount
_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::instanceCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceCount));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceCount(NS::UInteger instanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceCount_), instanceCount);
}

// property: instancedAccelerationStructures
_MTL_INLINE NS::Array* MTL::InstanceAccelerationStructureDescriptor::instancedAccelerationStructures() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(instancedAccelerationStructures));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstancedAccelerationStructures(const NS::Array* instancedAccelerationStructures)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstancedAccelerationStructures_), instancedAccelerationStructures);
}

// property: instanceDescriptorType
_MTL_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return Object::sendMessage<MTL::AccelerationStructureInstanceDescriptorType>(this, _MTL_PRIVATE_SEL(instanceDescriptorType));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorType_), instanceDescriptorType);
}

// property: motionTransformBuffer
_MTL_INLINE MTL::Buffer* MTL::InstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(motionTransformBuffer));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformBuffer(const MTL::Buffer* motionTransformBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBuffer_), motionTransformBuffer);
}

// property: motionTransformBufferOffset
_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::motionTransformBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformBufferOffset));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBufferOffset_), motionTransformBufferOffset);
}

// property: motionTransformCount
_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::motionTransformCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformCount));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformCount(NS::UInteger motionTransformCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformCount_), motionTransformCount);
}

// static method: descriptor
_MTL_INLINE MTL::InstanceAccelerationStructureDescriptor* MTL::InstanceAccelerationStructureDescriptor::descriptor()
{
    return Object::sendMessage<MTL::InstanceAccelerationStructureDescriptor*>(_MTL_PRIVATE_CLS(MTLInstanceAccelerationStructureDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

// static method: alloc
_MTL_INLINE MTL::IndirectInstanceAccelerationStructureDescriptor* MTL::IndirectInstanceAccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::IndirectInstanceAccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTLIndirectInstanceAccelerationStructureDescriptor));
}

// method: init
_MTL_INLINE MTL::IndirectInstanceAccelerationStructureDescriptor* MTL::IndirectInstanceAccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL::IndirectInstanceAccelerationStructureDescriptor>();
}

// property: instanceDescriptorBuffer
_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(instanceDescriptorBuffer));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(const MTL::Buffer* instanceDescriptorBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBuffer_), instanceDescriptorBuffer);
}

// property: instanceDescriptorBufferOffset
_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorBufferOffset));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBufferOffset_), instanceDescriptorBufferOffset);
}

// property: instanceDescriptorStride
_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorStride));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorStride_), instanceDescriptorStride);
}

// property: maxInstanceCount
_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::maxInstanceCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxInstanceCount));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMaxInstanceCount(NS::UInteger maxInstanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxInstanceCount_), maxInstanceCount);
}

// property: instanceCountBuffer
_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::instanceCountBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(instanceCountBuffer));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceCountBuffer(const MTL::Buffer* instanceCountBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceCountBuffer_), instanceCountBuffer);
}

// property: instanceCountBufferOffset
_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::instanceCountBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceCountBufferOffset));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceCountBufferOffset(NS::UInteger instanceCountBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceCountBufferOffset_), instanceCountBufferOffset);
}

// property: instanceDescriptorType
_MTL_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return Object::sendMessage<MTL::AccelerationStructureInstanceDescriptorType>(this, _MTL_PRIVATE_SEL(instanceDescriptorType));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorType_), instanceDescriptorType);
}

// property: motionTransformBuffer
_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(motionTransformBuffer));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformBuffer(const MTL::Buffer* motionTransformBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBuffer_), motionTransformBuffer);
}

// property: motionTransformBufferOffset
_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformBufferOffset));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBufferOffset_), motionTransformBufferOffset);
}

// property: maxMotionTransformCount
_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::maxMotionTransformCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxMotionTransformCount));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxMotionTransformCount_), maxMotionTransformCount);
}

// property: motionTransformCountBuffer
_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformCountBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(motionTransformCountBuffer));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformCountBuffer(const MTL::Buffer* motionTransformCountBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformCountBuffer_), motionTransformCountBuffer);
}

// property: motionTransformCountBufferOffset
_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformCountBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformCountBufferOffset));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformCountBufferOffset(NS::UInteger motionTransformCountBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformCountBufferOffset_), motionTransformCountBufferOffset);
}

// static method: descriptor
_MTL_INLINE MTL::IndirectInstanceAccelerationStructureDescriptor* MTL::IndirectInstanceAccelerationStructureDescriptor::descriptor()
{
    return Object::sendMessage<MTL::IndirectInstanceAccelerationStructureDescriptor*>(_MTL_PRIVATE_CLS(MTLIndirectInstanceAccelerationStructureDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

// property: size
_MTL_INLINE NS::UInteger MTL::AccelerationStructure::size() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(size));
}

// property: gpuResourceID
_MTL_INLINE MTL::ResourceID MTL::AccelerationStructure::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}
