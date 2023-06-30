//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// QuartzCore/CAMetalDrawable.hpp
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

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "../Metal/MTLPixelFormat.hpp"
#include "../Metal/MTLTexture.hpp"
#include <CoreGraphics/CGGeometry.h>

#include "CADefines.hpp"
#include "CAMetalDrawable.hpp"
#include "CAPrivate.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace CA
{

class MetalLayer : public NS::Referencing<MetalLayer>
{
public:
    static class MetalLayer* layer();

    MTL::Device*             device() const;
    void                     setDevice(MTL::Device* device);

    MTL::PixelFormat         pixelFormat() const;
    void                     setPixelFormat(MTL::PixelFormat pixelFormat);

    bool                     framebufferOnly() const;
    void                     setFramebufferOnly(bool framebufferOnly);

    CGSize                   drawableSize() const;
    void                     setDrawableSize(CGSize drawableSize);

    class MetalDrawable*     nextDrawable();
};
} // namespace CA

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
_CA_INLINE CA::MetalLayer* CA::MetalLayer::layer()
{
    return Object::sendMessage<CA::MetalLayer*>(_CA_PRIVATE_CLS(CAMetalLayer), _CA_PRIVATE_SEL(layer));
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_CA_INLINE MTL::Device* CA::MetalLayer::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _CA_PRIVATE_SEL(device));
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_CA_INLINE void CA::MetalLayer::setDevice(MTL::Device* device)
{
    return Object::sendMessage<void>(this, _CA_PRIVATE_SEL(setDevice_), device);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_CA_INLINE MTL::PixelFormat CA::MetalLayer::pixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this,
        _CA_PRIVATE_SEL(pixelFormat));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_CA_INLINE void CA::MetalLayer::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    return Object::sendMessage<void>(this, _CA_PRIVATE_SEL(setPixelFormat_),
        pixelFormat);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_CA_INLINE bool CA::MetalLayer::framebufferOnly() const
{
    return Object::sendMessage<bool>(this, _CA_PRIVATE_SEL(framebufferOnly));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_CA_INLINE void CA::MetalLayer::setFramebufferOnly(bool framebufferOnly)
{
    return Object::sendMessage<void>(this, _CA_PRIVATE_SEL(setFramebufferOnly_),
        framebufferOnly);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_CA_INLINE CGSize CA::MetalLayer::drawableSize() const
{
    return Object::sendMessage<CGSize>(this, _CA_PRIVATE_SEL(drawableSize));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_CA_INLINE void CA::MetalLayer::setDrawableSize(CGSize drawableSize)
{
    return Object::sendMessage<void>(this, _CA_PRIVATE_SEL(setDrawableSize_),
        drawableSize);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_CA_INLINE CA::MetalDrawable* CA::MetalLayer::nextDrawable()
{
    return Object::sendMessage<MetalDrawable*>(this,
        _CA_PRIVATE_SEL(nextDrawable));
}
