//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// MetalFX/MTLFXPrivate.hpp
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

#include "MTLFXDefines.hpp"

#include <objc/runtime.h>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#define _MTLFX_PRIVATE_CLS( symbol )                    ( Private::Class::s_k##symbol )
#define _MTLFX_PRIVATE_SEL( accessor )                  ( Private::Selector::s_k##accessor )

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#if defined( MTLFX_PRIVATE_IMPLEMENTATION )

#if defined( METALCPP_SYMBOL_VISIBILITY_HIDDEN )
#define _MTLFX_PRIVATE_VISIBILITY                       __attribute__( ( visibility("hidden" ) ) )
#else
#define _MTLFX_PRIVATE_VISIBILITY                       __attribute__( ( visibility("default" ) ) )
#endif // METALCPP_SYMBOL_VISIBILITY_HIDDEN

#define _MTLFX_PRIVATE_IMPORT                           __attribute__( ( weak_import ) )

#ifdef __OBJC__
#define _MTLFX_PRIVATE_OBJC_LOOKUP_CLASS( symbol )      ( ( __bridge void* ) objc_lookUpClass( #symbol ) )
#define _MTLFX_PRIVATE_OBJC_GET_PROTOCOL( symbol )      ( ( __bridge void* ) objc_getProtocol( #symbol ) )
#else
#define _MTLFX_PRIVATE_OBJC_LOOKUP_CLASS( symbol )      objc_lookUpClass(#symbol)
#define _MTLFX_PRIVATE_OBJC_GET_PROTOCOL( symbol )      objc_getProtocol(#symbol)
#endif // __OBJC__

#define _MTLFX_PRIVATE_DEF_CLS( symbol )                void* s_k##symbol _MTLFX_PRIVATE_VISIBILITY = _MTLFX_PRIVATE_OBJC_LOOKUP_CLASS( symbol )
#define _MTLFX_PRIVATE_DEF_PRO( symbol )                void* s_k##symbol _MTLFX_PRIVATE_VISIBILITY = _MTLFX_PRIVATE_OBJC_GET_PROTOCOL( symbol )
#define _MTLFX_PRIVATE_DEF_SEL( accessor, symbol )       SEL s_k##accessor _MTLFX_PRIVATE_VISIBILITY = sel_registerName( symbol )

#include <dlfcn.h>
#define MTLFX_DEF_FUNC( name, signature )               using Fn##name = signature; \
                                                        Fn##name name = reinterpret_cast< Fn##name >( dlsym( RTLD_DEFAULT, #name ) )

namespace MTLFX::Private
{
    template <typename _Type>

    inline _Type const LoadSymbol(const char* pSymbol)
    {
        const _Type* pAddress = static_cast<_Type*>(dlsym(RTLD_DEFAULT, pSymbol));

        return pAddress ? *pAddress : nullptr;
    }
} // MTLFX::Private

#if defined( __MAC_13_0 ) || defined( __MAC_14_0 ) || defined( __IPHONE_16_0 ) || defined( __IPHONE_17_0 ) || defined( __TVOS_16_0 ) || defined( __TVOS_17_0 )

#define _MTLFX_PRIVATE_DEF_STR( type, symbol )                                                                                  \
    _MTLFX_EXTERN type const                            MTLFX##symbol _MTLFX_PRIVATE_IMPORT;                                    \
    type const                                          MTLFX::symbol = ( nullptr != &MTLFX##symbol ) ? MTLFX##ssymbol : nullptr

#define _MTLFX_PRIVATE_DEF_CONST( type, symbol )                                                                                \
    _MTLFX_EXTERN type const                            MTLFX##ssymbol _MTLFX_PRIVATE_IMPORT;                                   \
    type const                                          MTLFX::symbol = (nullptr != &MTLFX##ssymbol) ? MTLFX##ssymbol : nullptr

#define _MTLFX_PRIVATE_DEF_WEAK_CONST( type, symbol )                                                                           \
    _MTLFX_EXTERN type const                            MTLFX##ssymbol;                                                         \
    type const                                          MTLFX::symbol = Private::LoadSymbol< type >( "MTLFX" #symbol )

#else

#define _MTLFX_PRIVATE_DEF_STR( type, symbol )                                                                                  \
    _MTLFX_EXTERN type const                            MTLFX##ssymbol;                                                         \
    type const                                          MTLFX::symbol = Private::LoadSymbol< type >( "MTLFX" #symbol )

#define _MTLFX_PRIVATE_DEF_CONST( type, symbol )                                                                                \
    _MTLFX_EXTERN type const                            MTLFX##ssymbol;                                                         \
    type const                                          MTLFX::symbol = Private::LoadSymbol< type >( "MTLFX" #symbol )

#define _MTLFX_PRIVATE_DEF_WEAK_CONST( type, symbol )   _MTLFX_PRIVATE_DEF_CONST( type, symbol )

#endif // defined( __MAC_13_0 ) || defined( __MAC_14_0 ) || defined( __IPHONE_16_0 ) || defined( __IPHONE_17_0 ) || defined( __TVOS_16_0 ) || defined( __TVOS_17_0 )

#else

#define _MTLFX_PRIVATE_DEF_CLS( symbol )                extern void* s_k##symbol
#define _MTLFX_PRIVATE_DEF_PRO( symbol )                extern void* s_k##symbol
#define _MTLFX_PRIVATE_DEF_SEL( accessor, symbol )      extern SEL s_k##accessor
#define _MTLFX_PRIVATE_DEF_STR( type, symbol )          extern type const MTLFX::symbol
#define _MTLFX_PRIVATE_DEF_CONST( type, symbol )        extern type const MTLFX::symbol
#define _MTLFX_PRIVATE_DEF_WEAK_CONST( type, symbol )   extern type const MTLFX::symbol

#endif // MTLFX_PRIVATE_IMPLEMENTATION

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTLFX
{
    namespace Private
    {
        namespace Class
        {
            _MTLFX_PRIVATE_DEF_CLS( MTLFXSpatialScalerDescriptor );
            _MTLFX_PRIVATE_DEF_CLS( MTLFXTemporalScalerDescriptor );
        } // Class
    } // Private
} // MTLFX

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTLFX
{
    namespace Private
    {
        namespace Protocol
        {
            _MTLFX_PRIVATE_DEF_PRO( MTLFXSpatialScaler );
            _MTLFX_PRIVATE_DEF_PRO( MTLFXTemporalScaler );
        } // Protocol
    } // Private
} // MTLFX

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace MTLFX
{
    namespace Private
    {
        namespace Selector
        {
            _MTLFX_PRIVATE_DEF_SEL( colorProcessingMode,
                                    "colorProcessingMode" );
            _MTLFX_PRIVATE_DEF_SEL( colorTexture,
                                    "colorTexture" );
            _MTLFX_PRIVATE_DEF_SEL( colorTextureFormat,
                                    "colorTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( colorTextureUsage,
                                    "colorTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( depthTexture,
                                    "depthTexture" );
            _MTLFX_PRIVATE_DEF_SEL( depthTextureFormat,
                                    "depthTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( depthTextureUsage,
                                    "depthTextureUsage" );                                    
            _MTLFX_PRIVATE_DEF_SEL( encodeToCommandBuffer_,
                                    "encodeToCommandBuffer:" );
            _MTLFX_PRIVATE_DEF_SEL( exposureTexture,
                                    "exposureTexture" );                                    
            _MTLFX_PRIVATE_DEF_SEL( fence,
                                    "fence" );
            _MTLFX_PRIVATE_DEF_SEL( inputContentHeight,
                                    "inputContentHeight" );
            _MTLFX_PRIVATE_DEF_SEL( inputContentMaxScale,
                                    "inputContentMaxScale" );
            _MTLFX_PRIVATE_DEF_SEL( inputContentMinScale,
                                    "inputContentMinScale" );
            _MTLFX_PRIVATE_DEF_SEL( inputContentWidth,
                                    "inputContentWidth" );
            _MTLFX_PRIVATE_DEF_SEL( inputHeight,
                                    "inputHeight" );
            _MTLFX_PRIVATE_DEF_SEL( inputWidth,
                                    "inputWidth" );
            _MTLFX_PRIVATE_DEF_SEL( isAutoExposureEnabled,
                                    "isAutoExposureEnabled" );
            _MTLFX_PRIVATE_DEF_SEL( isDepthReversed,
                                    "isDepthReversed" );
            _MTLFX_PRIVATE_DEF_SEL( isInputContentPropertiesEnabled,
                                    "isInputContentPropertiesEnabled" );
            _MTLFX_PRIVATE_DEF_SEL( jitterOffsetX,
                                    "jitterOffsetX" );
            _MTLFX_PRIVATE_DEF_SEL( jitterOffsetY,
                                    "jitterOffsetY" );
            _MTLFX_PRIVATE_DEF_SEL( motionTexture,
                                    "motionTexture" );
            _MTLFX_PRIVATE_DEF_SEL( motionTextureFormat,
                                    "motionTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( motionTextureUsage,
                                    "motionTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( motionVectorScaleX,
                                    "motionVectorScaleX" );
            _MTLFX_PRIVATE_DEF_SEL( motionVectorScaleY,
                                    "motionVectorScaleY" );
            _MTLFX_PRIVATE_DEF_SEL( newSpatialScalerWithDevice_,
                                    "newSpatialScalerWithDevice:" );
            _MTLFX_PRIVATE_DEF_SEL( newTemporalScalerWithDevice_,
                                    "newTemporalScalerWithDevice:" );
            _MTLFX_PRIVATE_DEF_SEL( outputHeight,
                                    "outputHeight" );
            _MTLFX_PRIVATE_DEF_SEL( outputTexture,
                                    "outputTexture" );
            _MTLFX_PRIVATE_DEF_SEL( outputTextureFormat,
                                    "outputTextureFormat" );
            _MTLFX_PRIVATE_DEF_SEL( outputTextureUsage,
                                    "outputTextureUsage" );
            _MTLFX_PRIVATE_DEF_SEL( outputWidth,
                                    "outputWidth" );
            _MTLFX_PRIVATE_DEF_SEL( preExposure,
                                    "preExposure" );
            _MTLFX_PRIVATE_DEF_SEL( reset,
                                    "reset" );
            _MTLFX_PRIVATE_DEF_SEL( setAutoExposureEnabled_,
                                    "setAutoExposureEnabled:" );
            _MTLFX_PRIVATE_DEF_SEL( setColorProcessingMode_,
                                    "setColorProcessingMode:" );
            _MTLFX_PRIVATE_DEF_SEL( setColorTexture_,
                                    "setColorTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setColorTextureFormat_,
                                    "setColorTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setDepthReversed_,
                                    "setDepthReversed:" );
            _MTLFX_PRIVATE_DEF_SEL( setDepthTexture_,
                                    "setDepthTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setDepthTextureFormat_,
                                    "setDepthTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setExposureTexture_,
                                    "setExposureTexture:" );                                    
            _MTLFX_PRIVATE_DEF_SEL( setFence_,
                                    "setFence:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentHeight_,
                                    "setInputContentHeight:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentMaxScale_,
                                    "setInputContentMaxScale:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentMinScale_,
                                    "setInputContentMinScale:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentPropertiesEnabled_,
                                    "setInputContentPropertiesEnabled:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputContentWidth_,
                                    "setInputContentWidth_:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputHeight_,
                                    "setInputHeight:" );
            _MTLFX_PRIVATE_DEF_SEL( setInputWidth_,
                                    "setInputWidth:" );
            _MTLFX_PRIVATE_DEF_SEL( setJitterOffsetX_,
                                    "setJitterOffsetX:" );
            _MTLFX_PRIVATE_DEF_SEL( setJitterOffsetY_,
                                    "setJitterOffsetY:" );
            _MTLFX_PRIVATE_DEF_SEL( setMotionTexture_,
                                    "setMotionTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setMotionTextureFormat_,
                                    "setMotionTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setMotionVectorScaleX_,
                                    "setMotionVectorScaleX:" );
            _MTLFX_PRIVATE_DEF_SEL( setMotionVectorScaleY_,
                                    "setMotionVectorScaleY:" );
            _MTLFX_PRIVATE_DEF_SEL( setOutputHeight_,
                                    "setOutputHeight:" );
            _MTLFX_PRIVATE_DEF_SEL( setOutputTexture_,
                                    "setOutputTexture:" );
            _MTLFX_PRIVATE_DEF_SEL( setOutputTextureFormat_,
                                    "setOutputTextureFormat:" );
            _MTLFX_PRIVATE_DEF_SEL( setOutputWidth_,
                                    "setOutputWidth:" );
            _MTLFX_PRIVATE_DEF_SEL( setPreExposure_,
                                    "setPreExposure:" );
            _MTLFX_PRIVATE_DEF_SEL( setReset_,
                                    "setReset:" );
            _MTLFX_PRIVATE_DEF_SEL( supportsDevice_,
                                    "supportsDevice:" );
        } // Selector
    } // Private
} // MTLFX

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
