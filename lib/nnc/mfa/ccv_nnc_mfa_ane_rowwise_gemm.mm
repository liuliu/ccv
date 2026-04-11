#include "ccv.h"
#include "ccv_nnc_mfa_ane_rowwise_internal.hpp"
#include "ccv_nnc_mfa_ane_rowwise_gemm.hpp"
#include "ccv_nnc_mfa_error.hpp"
#include "kernels/ANERowwiseTransformDescriptor.hpp"
#include "kernels/ANERowwiseTransformKernel.hpp"

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#if __has_include(<IOSurface/IOSurface.h>)
#import <IOSurface/IOSurface.h>
#else
#import <IOSurface/IOSurfaceRef.h>
#endif
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ccv::nnc;

typedef struct ccv_nnc_stream_context_s ccv_nnc_stream_context_t;

extern "C" {
void ccv_nnc_mfa_log_message(const char* message);
mtl_command_batch_t* ccv_nnc_stream_context_start_command_batch(ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_stream_context_finish_command_batch(ccv_nnc_stream_context_t* const stream_context, mtl_command_batch_t* command_batch);
id ccv_nnc_stream_context_finish_command_batch_encoding_and_return_mps_command_buffer(ccv_nnc_stream_context_t* const stream_context, mtl_command_batch_t* command_batch);
}

namespace {

constexpr MTLResourceOptions kSharedResourceOptions = MTLResourceStorageModeShared;
constexpr uint64_t kPrivateQuantCommitActivationElementsThreshold = (1ULL << 22);
constexpr uint32_t kANERowAlignment = 128;
constexpr size_t kANEScratchBufferAlignment = 16 * 1024;
constexpr size_t kANESurfaceCacheLimitBytes = 128 * 1024 * 1024;

struct CompiledProgram {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  CFTypeRef model;
  CFTypeRef x_name;
  CFTypeRef w_name;
  CFTypeRef y_name;

  ~CompiledProgram()
  {
    if (model)
      CFRelease(model);
    if (x_name)
      CFRelease(x_name);
    if (w_name)
      CFRelease(w_name);
    if (y_name)
      CFRelease(y_name);
  }
};

struct ProgramKey {
  uint32_t M;
  uint32_t N;
  uint32_t K;

  bool operator==(const ProgramKey& other) const noexcept
  {
    return M == other.M && N == other.N && K == other.K;
  }
};

struct ProgramKeyHash {
  size_t operator()(const ProgramKey& key) const noexcept
  {
    uint64_t hash = 1469598103934665603ULL;
    auto mix = [&](const uint64_t value) {
      hash ^= value;
      hash *= 1099511628211ULL;
    };
    mix((uint64_t)key.M);
    mix((uint64_t)key.N);
    mix((uint64_t)key.K);
    return (size_t)hash;
  }
};

std::unordered_map<ProgramKey, std::unique_ptr<CompiledProgram>, ProgramKeyHash> g_program_cache;

struct SharedScratch {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  size_t activation_surface_bytes;
  size_t weight_surface_bytes;
  size_t output_surface_bytes;
  size_t activation_scale_bytes;
  size_t activation_scale_capacity_bytes;
  IOSurfaceRef activation_surface;
  IOSurfaceRef weight_surface;
  IOSurfaceRef output_surface;
  MTL::Buffer* activation_surface_buffer;
  MTL::Buffer* weight_surface_buffer;
  MTL::Buffer* output_surface_buffer;
  MTL::Buffer* activation_scales;
};

static void destroy_shared_scratch(SharedScratch* const scratch);

struct CachedSurface {
  size_t width;
  size_t height;
  size_t alloc_bytes;
  CVPixelBufferRef pixel_buffer;
  IOSurfaceRef surface;
  MTL::Buffer* buffer;
  uint64_t last_used;
};

struct SurfaceCache {
  std::vector<std::unique_ptr<CachedSurface>> entries;
  size_t total_alloc_bytes;
  uint64_t lru_clock;
};

struct ANERowwiseGEMMCache {
  MTL::Device* device;
  SharedScratch scratch;
  SurfaceCache activation_surface_cache;
  SurfaceCache weight_surface_cache;
  SurfaceCache output_surface_cache;
};

static size_t align_up(const size_t value, const size_t alignment) noexcept
{
  return (value + alignment - 1) & ~(alignment - 1);
}

static inline id<MTLCommandBuffer> bridge_command_buffer(mtl_command_buffer_t* const command_buffer)
{
  return (__bridge id<MTLCommandBuffer>)(void*)command_buffer;
}

static inline id<MTLBuffer> bridge_buffer(MTL::Buffer* const buffer)
{
  return (__bridge id<MTLBuffer>)(void*)buffer;
}

static void log_ane_rowwise_error(ccv_nnc_mfa_context_t* const context, const std::string& error)
{
  if (error.empty())
    return;
  if (METAL_LOG_LEVEL(context) >= 1)
    ccv_nnc_mfa_log_message(error.c_str());
}

static std::string describe_nserror(NSError* const error)
{
  if (!error)
    return "unknown error";
  return std::string([[error description] UTF8String]);
}

static size_t bytes_per_ml_datatype(const MLMultiArrayDataType dt)
{
  switch (dt) {
    case MLMultiArrayDataTypeDouble:
      return sizeof(double);
    case MLMultiArrayDataTypeFloat32:
      return sizeof(float);
    case MLMultiArrayDataTypeFloat16:
      return sizeof(uint16_t);
    case MLMultiArrayDataTypeInt32:
      return sizeof(int32_t);
    case MLMultiArrayDataTypeInt8:
      return sizeof(int8_t);
    default:
      return 0;
  }
}

static OSType pixel_format_for_ml_datatype(const MLMultiArrayDataType dt)
{
  switch (dt) {
    case MLMultiArrayDataTypeInt8:
      return kCVPixelFormatType_OneComponent8;
    case MLMultiArrayDataTypeFloat16:
      return kCVPixelFormatType_OneComponent16Half;
    default:
      return 0;
  }
}

static NSArray<NSNumber*>* make_shape(const NSInteger d0, const NSInteger d1, const NSInteger d2, const NSInteger d3)
{
  return @[@(d0), @(d1), @(d2), @(d3)];
}

static bool output_backing_key_supported(void)
{
  return [[MLPredictionOptions class] instancesRespondToSelector:NSSelectorFromString(@"setOutputBackings:")];
}

static CVPixelBufferRef create_pixel_buffer_for_dimensions(
    const size_t width,
    const size_t height,
    const MLMultiArrayDataType dt,
    std::string* const error_out)
{
  const size_t bytes_per_element = bytes_per_ml_datatype(dt);
  const OSType pixel_format = pixel_format_for_ml_datatype(dt);
  if (!width || !height || !bytes_per_element || !pixel_format) {
    if (error_out)
      *error_out = "unsupported pixel buffer configuration";
    return nullptr;
  }
  NSDictionary* const attrs = @{
    (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
    (id)kCVPixelBufferMetalCompatibilityKey : @YES,
  };
  CVPixelBufferRef pixel_buffer = nullptr;
  const CVReturn rc = CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixel_format, (__bridge CFDictionaryRef)attrs, &pixel_buffer);
  if (rc != kCVReturnSuccess || !pixel_buffer) {
    if (error_out)
      *error_out = "CVPixelBufferCreate failed (" + std::to_string((int)rc) + ")";
    return nullptr;
  }
  return pixel_buffer;
}

static CVPixelBufferRef create_pixel_buffer_for_surface(
    IOSurfaceRef const surface,
    NSArray<NSNumber*>* const shape,
    const MLMultiArrayDataType dt,
    std::string* const error_out)
{
  if (!surface) {
    if (error_out)
      *error_out = "IOSurface is not available for pixel buffer creation";
    return nullptr;
  }
  const size_t bytes_per_element = bytes_per_ml_datatype(dt);
  const OSType pixel_format = pixel_format_for_ml_datatype(dt);
  if (!shape || shape.count == 0 || !bytes_per_element || !pixel_format) {
    if (error_out)
      *error_out = "unsupported pixel buffer surface configuration";
    return nullptr;
  }
  const size_t width = (size_t)shape.lastObject.unsignedLongLongValue;
  size_t height = 1;
  for (NSUInteger i = 0; i + 1 < shape.count; ++i)
    height *= shape[i].unsignedIntegerValue;
  NSDictionary* const attrs = @{
    (id)kCVPixelBufferWidthKey : @(width),
    (id)kCVPixelBufferHeightKey : @(height),
    (id)kCVPixelBufferPixelFormatTypeKey : @(pixel_format),
    (id)kCVPixelBufferBytesPerRowAlignmentKey : @(width * bytes_per_element),
    (id)kCVPixelBufferMetalCompatibilityKey : @YES,
  };
  CVPixelBufferRef pixel_buffer = nullptr;
  const CVReturn rc = CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, surface, (__bridge CFDictionaryRef)attrs, &pixel_buffer);
  if (rc != kCVReturnSuccess || !pixel_buffer) {
    if (error_out)
      *error_out = "CVPixelBufferCreateWithIOSurface failed (" + std::to_string((int)rc) + ")";
    return nullptr;
  }
  return pixel_buffer;
}

static MLMultiArray* create_multiarray_with_pixel_buffer(
    CVPixelBufferRef const pixel_buffer,
    NSArray<NSNumber*>* const shape,
    std::string* const error_out)
{
  @try {
    return [[[MLMultiArray alloc] initWithPixelBuffer:pixel_buffer shape:shape] autorelease];
  } @catch (NSException* exception) {
    if (error_out)
      *error_out = exception.reason ? std::string(exception.reason.UTF8String) : "initWithPixelBuffer failed";
    return nil;
  }
}

static MTL::Buffer* create_surface_buffer(MTL::Device* const device, IOSurfaceRef const surface, const size_t bytes)
{
  if (!surface)
    return nullptr;
  void* const base_address = IOSurfaceGetBaseAddress(surface);
  return device->newBuffer(base_address, bytes, kSharedResourceOptions, nil);
}

static void destroy_cached_surface(CachedSurface* const entry)
{
  if (!entry)
    return;
  if (entry->pixel_buffer)
    CFRelease(entry->pixel_buffer);
  if (entry->surface)
    CFRelease(entry->surface);
  if (entry->buffer)
    entry->buffer->release();
  entry->pixel_buffer = nullptr;
  entry->surface = nullptr;
  entry->buffer = nullptr;
}

static void destroy_surface_cache(SurfaceCache* const cache)
{
  if (!cache)
    return;
  for (auto& entry : cache->entries)
    destroy_cached_surface(entry.get());
  cache->entries.clear();
  cache->total_alloc_bytes = 0;
  cache->lru_clock = 0;
}

static void evict_surface_cache_entries(SurfaceCache* const cache, CachedSurface* const keep_entry)
{
  while (cache->total_alloc_bytes > kANESurfaceCacheLimitBytes) {
    auto victim_it = cache->entries.end();
    for (auto it = cache->entries.begin(); it != cache->entries.end(); ++it) {
      if (it->get() == keep_entry)
        continue;
      if (victim_it == cache->entries.end() || (*it)->last_used < (*victim_it)->last_used)
        victim_it = it;
    }
    if (victim_it == cache->entries.end())
      break;
    cache->total_alloc_bytes -= (*victim_it)->alloc_bytes;
    destroy_cached_surface(victim_it->get());
    cache->entries.erase(victim_it);
  }
}

static CachedSurface* find_or_create_cached_surface(
    SurfaceCache* const cache,
    MTL::Device* const device,
    const size_t width,
    const size_t requested_height,
    const MLMultiArrayDataType dt,
    std::string* const error_out)
{
  for (auto& entry : cache->entries)
    if (entry->width == width && entry->height == requested_height) {
      entry->last_used = ++cache->lru_clock;
      return entry.get();
    }

  std::unique_ptr<CachedSurface> replacement(new CachedSurface {});
  replacement->width = width;
  replacement->height = requested_height;
  replacement->last_used = ++cache->lru_clock;
  replacement->pixel_buffer = create_pixel_buffer_for_dimensions(width, requested_height, dt, error_out);
  replacement->surface = replacement->pixel_buffer ? (IOSurfaceRef)CFRetain(CVPixelBufferGetIOSurface(replacement->pixel_buffer)) : nullptr;
  replacement->alloc_bytes = replacement->surface ? IOSurfaceGetAllocSize(replacement->surface) : 0;
  replacement->buffer = create_surface_buffer(device, replacement->surface, replacement->alloc_bytes);
  if (!replacement->pixel_buffer || !replacement->surface || !replacement->buffer) {
    if (error_out && error_out->empty())
      *error_out = "failed to allocate cached CoreML rowwise surface";
    destroy_cached_surface(replacement.get());
    return nullptr;
  }

  cache->total_alloc_bytes += replacement->alloc_bytes;
  cache->entries.emplace_back(std::move(replacement));
  CachedSurface* const created = cache->entries.back().get();
  evict_surface_cache_entries(cache, created);
  return created;
}

static void destroy_shared_scratch(SharedScratch* const scratch)
{
  if (!scratch)
    return;
  if (scratch->activation_scales)
    scratch->activation_scales->release();
  scratch->M = 0;
  scratch->N = 0;
  scratch->K = 0;
  scratch->activation_surface_bytes = 0;
  scratch->weight_surface_bytes = 0;
  scratch->output_surface_bytes = 0;
  scratch->activation_scale_bytes = 0;
  scratch->activation_scale_capacity_bytes = 0;
  scratch->activation_surface = nullptr;
  scratch->weight_surface = nullptr;
  scratch->output_surface = nullptr;
  scratch->activation_surface_buffer = nullptr;
  scratch->weight_surface_buffer = nullptr;
  scratch->output_surface_buffer = nullptr;
  scratch->activation_scales = nullptr;
}

static float choose_model_scale(const uint32_t K)
{
  const float scale = 1.0f / std::sqrt((float)K);
  uint16_t scale_bits;
  float rounded_scale;
  ccv_float_to_half_precision(&scale, &scale_bits, 1);
  ccv_half_precision_to_float(&scale_bits, &rounded_scale, 1);
  return rounded_scale;
}

static uint32_t pad_ane_rows(const uint32_t rows)
{
  return (uint32_t)align_up(rows, kANERowAlignment);
}

static uint32_t rowwise_batch_dimension(const ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  return params.batch_dimension ? params.batch_dimension : 1;
}

static uint32_t rowwise_total_rows(const ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  return params.M * rowwise_batch_dimension(params);
}

static uint32_t rowwise_padded_total_rows(const ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  return pad_ane_rows(rowwise_total_rows(params));
}

static NSString* mil_header()
{
  return @"program(1.3)\n"
         @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         @"{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
         @"{\"coremltools-version\", \"9.0\"}})]\n"
         @"{\n";
}

static NSString* mil_fp16_literal(const float value)
{
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "fp16(%a)", value);
  return [NSString stringWithUTF8String:buffer];
}

static NSString* shape_list_for_metadata(NSArray<NSNumber*>* const shape)
{
  NSMutableString* const s = [NSMutableString stringWithString:@"["];
  for (NSUInteger i = 0; i < shape.count; ++i)
    [s appendFormat:@"%@%@", shape[i], (i + 1 == shape.count) ? @"]" : @", "];
  return s;
}

static NSString* formatted_shape_from_tensor_shape(NSArray<NSNumber*>* const shape)
{
  NSMutableString* const s = [NSMutableString string];
  for (NSUInteger i = 0; i < shape.count; ++i) {
    [s appendFormat:@"%@", shape[i]];
    if (i + 1 != shape.count)
      [s appendString:@" x "];
  }
  return s;
}

static NSDictionary* schema_entry_for_name(NSString* const name, const char* const dtype, NSArray<NSNumber*>* const shape)
{
  return @{
    @"hasShapeFlexibility" : @"0",
    @"isOptional" : @"0",
    @"dataType" : [NSString stringWithUTF8String:dtype],
    @"shortDescription" : @"",
    @"shape" : shape_list_for_metadata(shape),
    @"formattedType" : [NSString stringWithFormat:@"MultiArray (%s %@)", dtype, formatted_shape_from_tensor_shape(shape)],
    @"name" : name,
    @"type" : @"MultiArray"
  };
}

static void append_uvarint(NSMutableData* const data, uint64_t value)
{
  do {
    uint8_t byte = (uint8_t)(value & 0x7f);
    value >>= 7;
    if (value)
      byte |= 0x80;
    [data appendBytes:&byte length:1];
  } while (value);
}

static void append_coremldata_feature_entry(
    NSMutableData* const payload,
    const uint8_t tag,
    NSString* const name,
    NSArray<NSNumber*>* const shape,
    const uint64_t dtype_code)
{
  NSData* const name_data = [name dataUsingEncoding:NSASCIIStringEncoding];
  NSMutableData* const dims = [NSMutableData data];
  for (NSNumber* dim in shape)
    append_uvarint(dims, dim.unsignedLongLongValue);

  NSMutableData* const nested = [NSMutableData data];
  const uint8_t dims_tag = 0x0a;
  const uint8_t dims_len = (uint8_t)dims.length;
  [nested appendBytes:&dims_tag length:1];
  [nested appendBytes:&dims_len length:1];
  [nested appendData:dims];
  const uint8_t dtype_tag = 0x10;
  [nested appendBytes:&dtype_tag length:1];
  append_uvarint(nested, dtype_code);

  NSMutableData* const inner = [NSMutableData data];
  const uint8_t inner_tag = 0x2a;
  const uint8_t inner_len = (uint8_t)nested.length;
  [inner appendBytes:&inner_tag length:1];
  [inner appendBytes:&inner_len length:1];
  [inner appendData:nested];

  NSMutableData* const body = [NSMutableData data];
  const uint8_t name_tag = 0x0a;
  const uint8_t name_len = (uint8_t)name_data.length;
  [body appendBytes:&name_tag length:1];
  [body appendBytes:&name_len length:1];
  [body appendData:name_data];
  const uint8_t shape_tag = 0x1a;
  const uint8_t shape_len = (uint8_t)inner.length;
  [body appendBytes:&shape_tag length:1];
  [body appendBytes:&shape_len length:1];
  [body appendData:inner];

  const uint8_t body_len = (uint8_t)body.length;
  [payload appendBytes:&tag length:1];
  [payload appendBytes:&body_len length:1];
  [payload appendData:body];
}

static NSString* generate_dynamic_i8_two_input_matmul_tx_scaled_mil(
    const uint32_t K,
    const uint32_t N,
    const uint32_t padded_M,
    const float dq_scale)
{
  NSMutableString* const mil = [NSMutableString string];
  [mil appendString:mil_header()];
  [mil appendFormat:@"    func main<ios19>(tensor<int8, [1, 1, %u, %u]> w, tensor<int8, [1, 1, %u, %u]> x) {\n", N, K, K, padded_M];
  [mil appendFormat:@"        fp16 dq_scale = const()[name = string(\"dq_scale\"), val = %@];\n", mil_fp16_literal(dq_scale)];
  [mil appendFormat:@"        tensor<fp16, [1,1,%u,%u]> xh = dequantize(input = x, scale = dq_scale)[name = string(\"xh\")];\n", K, padded_M];
  [mil appendFormat:@"        tensor<fp16, [1,1,%u,%u]> wh = dequantize(input = w, scale = dq_scale)[name = string(\"wh\")];\n", N, K];
  [mil appendString:@"        bool mm_transpose_x_0 = const()[name = string(\"mm_transpose_x_0\"), val = bool(false)];\n"];
  [mil appendString:@"        bool mm_transpose_y_0 = const()[name = string(\"mm_transpose_y_0\"), val = bool(false)];\n"];
  [mil appendFormat:@"        tensor<fp16, [1, 1, %u, %u]> mm = matmul(transpose_x = mm_transpose_x_0, transpose_y = mm_transpose_y_0, x = wh, y = xh)[name = string(\"mm\")];\n", N, padded_M];
  [mil appendString:@"    } -> (mm);\n}\n"];
  return mil;
}

static NSString* build_manifest_json(const NSString* const model_path, const NSString* const weight_path)
{
  NSString* const model_id = [[NSUUID UUID] UUIDString];
  NSString* const weight_id = [[NSUUID UUID] UUIDString];
  NSDictionary* const manifest = @{
    @"fileFormatVersion": @"1.0.0",
    @"itemInfoEntries": @{
      model_id: @{
        @"author": @"com.apple.CoreML",
        @"description": @"CoreML Model Specification",
        @"name": [model_path.lastPathComponent stringByDeletingPathExtension],
        @"path": model_path,
      },
      weight_id: @{
        @"author": @"com.apple.CoreML",
        @"description": @"CoreML Model Weights",
        @"name": @"weights",
        @"path": weight_path,
      },
    },
    @"rootModelIdentifier": model_id,
  };
  NSData* const data = [NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingPrettyPrinted error:nil];
  return [[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] autorelease];
}

static bool write_text_file(NSString* const text, NSString* const path, std::string* const error_out)
{
  NSData* const data = [text dataUsingEncoding:NSUTF8StringEncoding];
  if (data && [data writeToFile:path atomically:YES])
    return true;
  if (error_out)
    *error_out = "failed to write " + std::string(path.UTF8String);
  return false;
}

static NSMutableDictionary* default_metadata_root(void)
{
  return [@{
    @"metadataOutputVersion" : @"3.0",
    @"outputSchema" : @[],
    @"modelParameters" : @[],
    @"specificationVersion" : @10,
    @"mlProgramOperationTypeHistogram" : @{ @"Ios19.dequantize" : @2, @"Ios18.matmul" : @1 },
    @"computePrecision" : @"Float16",
    @"isUpdatable" : @"0",
    @"stateSchema" : @[],
    @"availability" : @{
      @"macOS" : @"16.0",
      @"tvOS" : @"19.0",
      @"visionOS" : @"3.0",
      @"watchOS" : @"12.0",
      @"iOS" : @"19.0",
      @"macCatalyst" : @"19.0"
    },
    @"modelType" : @{ @"name" : @"MLModelType_mlProgram" },
    @"userDefinedMetadata" : @{
      @"com.github.apple.coremltools.version" : @"9.0",
      @"com.github.apple.coremltools.source" : @"milinternal",
      @"com.github.apple.coremltools.conversion_date" : @"2026-04-08"
    },
    @"inputSchema" : @[],
    @"generatedClassName" : @"w_a",
    @"method" : @"predict"
  } mutableCopy];
}

static NSString* build_metadata_json(const uint32_t K, const uint32_t N, const uint32_t padded_M)
{
  NSMutableDictionary* const root = default_metadata_root();
  NSArray<NSNumber*>* const x_shape = make_shape(1, 1, K, padded_M);
  NSArray<NSNumber*>* const w_shape = make_shape(1, 1, N, K);
  NSArray<NSNumber*>* const y_shape = make_shape(1, 1, N, padded_M);
  NSString* const output_name = @"mm";

  root[@"metadataOutputVersion"] = @"3.0";
  root[@"modelParameters"] = @[];
  root[@"stateSchema"] = @[];
  root[@"method"] = @"predict";
  root[@"mlProgramOperationTypeHistogram"] = @{ @"Ios19.dequantize" : @2, @"Ios18.matmul" : @1 };
  root[@"inputSchema"] = @[
    schema_entry_for_name(@"x", "Int8", x_shape),
    schema_entry_for_name(@"w", "Int8", w_shape),
  ];
  root[@"outputSchema"] = @[
    schema_entry_for_name(output_name, "Float16", y_shape),
  ];
  root[@"generatedClassName"] = @"w_a";

  NSData* const json_data = [NSJSONSerialization dataWithJSONObject:@[root] options:NSJSONWritingPrettyPrinted error:nil];
  return [[[NSString alloc] initWithData:json_data encoding:NSUTF8StringEncoding] autorelease];
}

static const uint8_t kManualCoreMLDataHeader[0x53] = {
  0xf6, 0x01, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x67, 0x65, 0x6e, 0x65, 0x72, 0x69, 0x63, 0x0a, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00
};

static const uint8_t kManualCoreMLDataTrailer[16] = {
  0xf6, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

static const uint8_t kCoreMLToolsCoreMLDataMetadataField[164] = {
  0xa2, 0x06, 0xa0, 0x01, 0xa2, 0x06, 0x3a, 0x0a, 0x2c, 0x63, 0x6f, 0x6d, 0x2e, 0x67, 0x69, 0x74,
  0x68, 0x75, 0x62, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x65, 0x2e, 0x63, 0x6f, 0x72, 0x65, 0x6d, 0x6c,
  0x74, 0x6f, 0x6f, 0x6c, 0x73, 0x2e, 0x63, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e,
  0x5f, 0x64, 0x61, 0x74, 0x65, 0x12, 0x0a, 0x32, 0x30, 0x32, 0x36, 0x2d, 0x30, 0x34, 0x2d, 0x30,
  0x38, 0xa2, 0x06, 0x32, 0x0a, 0x23, 0x63, 0x6f, 0x6d, 0x2e, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62,
  0x2e, 0x61, 0x70, 0x70, 0x6c, 0x65, 0x2e, 0x63, 0x6f, 0x72, 0x65, 0x6d, 0x6c, 0x74, 0x6f, 0x6f,
  0x6c, 0x73, 0x2e, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x12, 0x0b, 0x6d, 0x69, 0x6c, 0x69, 0x6e,
  0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0xa2, 0x06, 0x2b, 0x0a, 0x24, 0x63, 0x6f, 0x6d, 0x2e, 0x67,
  0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x61, 0x70, 0x70, 0x6c, 0x65, 0x2e, 0x63, 0x6f, 0x72, 0x65,
  0x6d, 0x6c, 0x74, 0x6f, 0x6f, 0x6c, 0x73, 0x2e, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x12,
  0x03, 0x39, 0x2e, 0x30
};

static NSData* synthesized_coremldata(const uint32_t K, const uint32_t N, const uint32_t padded_M)
{
  NSMutableData* const payload = [NSMutableData data];
  append_coremldata_feature_entry(payload, 0x0a, @"x", make_shape(1, 1, K, padded_M), 0x20008);
  append_coremldata_feature_entry(payload, 0x0a, @"w", make_shape(1, 1, N, K), 0x20008);
  append_coremldata_feature_entry(payload, 0x52, @"mm", make_shape(1, 1, N, padded_M), 0x10010);
  [payload appendBytes:kCoreMLToolsCoreMLDataMetadataField length:sizeof(kCoreMLToolsCoreMLDataMetadataField)];

  NSMutableData* const out = [NSMutableData dataWithBytes:kManualCoreMLDataHeader length:sizeof(kManualCoreMLDataHeader)];
  const uint64_t payload_len = (uint64_t)payload.length;
  uint8_t payload_len_le[8];
  for (int i = 0; i < 8; ++i)
    payload_len_le[i] = (uint8_t)((payload_len >> (8 * i)) & 0xff);
  [out replaceBytesInRange:NSMakeRange(0x4b, 8) withBytes:payload_len_le];
  [out appendData:payload];
  [out appendBytes:kManualCoreMLDataTrailer length:sizeof(kManualCoreMLDataTrailer)];
  return out;
}

static void select_two_input_names(
    NSDictionary<NSString*, MLFeatureDescription*>* const features,
    NSString** const x_out,
    NSString** const w_out)
{
  NSString* x = @"x";
  NSString* w = @"w";
  if (features && features.count >= 2) {
    NSMutableArray<NSString*>* const keys = [NSMutableArray array];
    for (NSString* key in features) {
      MLFeatureDescription* const desc = features[key];
      if (desc.type == MLFeatureTypeMultiArray)
        [keys addObject:key];
    }
    if (keys.count >= 2) {
      x = [keys containsObject:@"x"] ? @"x" : keys[0];
      w = [keys containsObject:@"w"] ? @"w" : ([keys[0] isEqualToString:x] ? keys[1] : keys[0]);
    }
  }
  if (x_out)
    *x_out = x;
  if (w_out)
    *w_out = w;
}

static size_t rowwise_8i_scale_offset(const uint32_t rows, const uint32_t cols)
{
  return align_up((size_t)rows * cols * sizeof(int8_t), 128);
}

static ANERowwiseGEMMCache* get_or_create_cache(ccv_nnc_mfa_context_t* const context, std::string* const error_out)
{
  (void)error_out;
  if (ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context))
    return (ANERowwiseGEMMCache*)ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context);
  auto* const cache = new ANERowwiseGEMMCache();
  cache->device = ccv_nnc_mfa_context_device(context);
  ccv_nnc_mfa_context_set_ane_rowwise_gemm_cache(context, cache);
  return cache;
}

static bool ensure_shared_scratch(
    ANERowwiseGEMMCache* const cache,
    const ccv_nnc_mfa_ane_rowwise_gemm_params_t params,
    std::string* const error_out)
{
  const uint32_t padded_M = rowwise_padded_total_rows(params);
  const size_t activation_surface_bytes = (size_t)params.K * padded_M * sizeof(int8_t);
  const size_t weight_surface_bytes = (size_t)params.N * params.K * sizeof(int8_t);
  const size_t output_surface_bytes = (size_t)params.N * padded_M * sizeof(uint16_t);
  const size_t activation_scale_bytes = (size_t)padded_M * sizeof(uint16_t);
  SharedScratch& scratch = cache->scratch;
  const bool scratch_shape_matches = (scratch.M == padded_M && scratch.N == params.N && scratch.K == params.K);
  if (scratch_shape_matches &&
      scratch.activation_surface && scratch.weight_surface && scratch.output_surface &&
      scratch.activation_surface_buffer && scratch.weight_surface_buffer &&
      scratch.output_surface_buffer && scratch.activation_scales &&
      scratch.activation_scale_capacity_bytes >= activation_scale_bytes)
    return true;
  CachedSurface* const activation_surface_entry =
    find_or_create_cached_surface(&cache->activation_surface_cache, cache->device, padded_M, params.K, MLMultiArrayDataTypeInt8, error_out);
  CachedSurface* const weight_surface_entry =
    find_or_create_cached_surface(&cache->weight_surface_cache, cache->device, params.K, params.N, MLMultiArrayDataTypeInt8, error_out);
  CachedSurface* const output_surface_entry =
    find_or_create_cached_surface(&cache->output_surface_cache, cache->device, padded_M, params.N, MLMultiArrayDataTypeFloat16, error_out);
  scratch.activation_surface = activation_surface_entry ? activation_surface_entry->surface : nullptr;
  scratch.weight_surface = weight_surface_entry ? weight_surface_entry->surface : nullptr;
  scratch.output_surface = output_surface_entry ? output_surface_entry->surface : nullptr;
  scratch.activation_surface_buffer = activation_surface_entry ? activation_surface_entry->buffer : nullptr;
  scratch.weight_surface_buffer = weight_surface_entry ? weight_surface_entry->buffer : nullptr;
  scratch.output_surface_buffer = output_surface_entry ? output_surface_entry->buffer : nullptr;
  if (!scratch.activation_scales || scratch.activation_scale_capacity_bytes < activation_scale_bytes) {
    if (scratch.activation_scales) {
      scratch.activation_scales->release();
      scratch.activation_scales = nullptr;
    }
    const size_t aligned_activation_scale_bytes = align_up(activation_scale_bytes, kANEScratchBufferAlignment);
    scratch.activation_scales = cache->device->newBuffer(aligned_activation_scale_bytes, kSharedResourceOptions);
    scratch.activation_scale_capacity_bytes = scratch.activation_scales ? aligned_activation_scale_bytes : 0;
  }
  scratch.M = padded_M;
  scratch.N = params.N;
  scratch.K = params.K;
  scratch.activation_surface_bytes = activation_surface_bytes;
  scratch.weight_surface_bytes = weight_surface_bytes;
  scratch.output_surface_bytes = output_surface_bytes;
  scratch.activation_scale_bytes = activation_scale_bytes;
  if (!scratch.activation_surface || !scratch.weight_surface || !scratch.output_surface ||
      !scratch.activation_surface_buffer || !scratch.weight_surface_buffer ||
      !scratch.output_surface_buffer || !scratch.activation_scales) {
    if (error_out)
      *error_out = "failed to allocate CoreML rowwise scratch";
    destroy_surface_cache(&cache->activation_surface_cache);
    destroy_surface_cache(&cache->weight_surface_cache);
    destroy_surface_cache(&cache->output_surface_cache);
    destroy_shared_scratch(&scratch);
    return false;
  }
  return true;
}

static PipelineValue<ANERowwiseTransformKernel>* find_transform_pipeline(
    ccv_nnc_mfa_context_t* const context,
    const ccv_nnc_mfa_ane_rowwise_gemm_params_t params)
{
  ANERowwiseTransformDescriptor descriptor;
  if (params.data_type == MTL::DataTypeHalf) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  } else {
    CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeBFloat);
    descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
  }
  descriptor.M = params.M;
  descriptor.paddedM = rowwise_padded_total_rows(params);
  descriptor.batchDimension = rowwise_batch_dimension(params);
  descriptor.N = params.N;
  descriptor.K = params.K;
  descriptor.batchStrideA = params.batch_stride_a;
  descriptor.batchStrideC = params.batch_stride_c;
  return ccv_nnc_mfa_prepare_ane_rowwise_transform(context, descriptor);
}

static std::unique_ptr<CompiledProgram> compile_program(
    const ccv_nnc_mfa_ane_rowwise_gemm_params_t params,
    std::string* const error_out)
{
  @autoreleasepool {
    const uint32_t padded_M = rowwise_padded_total_rows(params);
    const float model_scale = choose_model_scale(params.K);
    NSString* const temp_directory = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.mlmodelc", [[NSUUID UUID] UUIDString]]];
    NSFileManager* const file_manager = [NSFileManager defaultManager];
    NSError* error = nil;
    if (![file_manager createDirectoryAtPath:[temp_directory stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:&error]) {
      if (error_out)
        *error_out = "failed to create manual .mlmodelc: " + describe_nserror(error);
      return {};
    }
    NSData* const coremldata = synthesized_coremldata(params.K, params.N, padded_M);
    if (![coremldata writeToFile:[temp_directory stringByAppendingPathComponent:@"coremldata.bin"] atomically:YES]) {
      if (error_out)
        *error_out = "failed to write coremldata.bin";
      return {};
    }
    if (!write_text_file(generate_dynamic_i8_two_input_matmul_tx_scaled_mil(params.K, params.N, padded_M, model_scale),
                         [temp_directory stringByAppendingPathComponent:@"model.mil"], error_out))
      return {};
    NSString* const metadata_json = build_metadata_json(params.K, params.N, padded_M);
    if (!write_text_file(metadata_json,
                         [temp_directory stringByAppendingPathComponent:@"metadata.json"], error_out))
      return {};
    NSString* const manifest_json = build_manifest_json(@"model.mil", @"weights");
    if (!write_text_file(manifest_json,
                         [temp_directory stringByAppendingPathComponent:@"Manifest.json"], error_out))
      return {};
    if (![[NSData data] writeToFile:[temp_directory stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES]) {
      if (error_out)
        *error_out = "failed to write empty weight.bin";
      return {};
    }
    MLModelConfiguration* const cfg = [[[MLModelConfiguration alloc] init] autorelease];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    NSURL* const model_dir_url = [NSURL fileURLWithPath:temp_directory];
    MLModel* const model = [MLModel modelWithContentsOfURL:model_dir_url configuration:cfg error:&error];
    if (!model) {
      if (error_out)
        *error_out = "modelWithContentsOfURL failed: " + describe_nserror(error);
      return {};
    }
    NSDictionary<NSString*, MLFeatureDescription*>* const in_desc = model.modelDescription.inputDescriptionsByName;
    NSDictionary<NSString*, MLFeatureDescription*>* const out_desc = model.modelDescription.outputDescriptionsByName;
    if (in_desc.count < 2 || out_desc.count < 1) {
      if (error_out)
        *error_out = "invalid model feature counts";
      return {};
    }
    NSString* x_name = @"x";
    NSString* w_name = @"w";
    select_two_input_names(in_desc, &x_name, &w_name);
    NSString* const y_name = out_desc.allKeys[0];
    MLFeatureDescription* const x_desc = in_desc[x_name];
    MLFeatureDescription* const w_desc = in_desc[w_name];
    MLFeatureDescription* const y_desc = out_desc[y_name];
    if (!x_desc.multiArrayConstraint || !w_desc.multiArrayConstraint || !y_desc.multiArrayConstraint) {
      if (error_out)
        *error_out = "missing multiarray constraints on loaded CoreML model";
      return {};
    }
    if (x_desc.multiArrayConstraint.dataType != MLMultiArrayDataTypeInt8 ||
        w_desc.multiArrayConstraint.dataType != MLMultiArrayDataTypeInt8 ||
        y_desc.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat16) {
      if (error_out)
        *error_out = "unexpected CoreML model I/O data types";
      return {};
    }
    if (![file_manager removeItemAtPath:temp_directory error:&error]) {
      if (error_out)
        *error_out = "failed to remove loaded manual .mlmodelc: " + describe_nserror(error);
      return {};
    }
    std::unique_ptr<CompiledProgram> program(new CompiledProgram {
      padded_M,
      params.N,
      params.K,
      CFBridgingRetain(model),
      CFBridgingRetain(x_name),
      CFBridgingRetain(w_name),
      CFBridgingRetain(y_name),
    });
    return program;
  }
}

static CompiledProgram* find_or_create_program(
    const ccv_nnc_mfa_ane_rowwise_gemm_params_t params,
    std::string* const error_out)
{
  const ProgramKey key = {
    .M = rowwise_padded_total_rows(params),
    .N = params.N,
    .K = params.K,
  };
  auto it = g_program_cache.find(key);
  if (it != g_program_cache.end())
    return it->second.get();
  std::unique_ptr<CompiledProgram> program = compile_program(params, error_out);
  if (!program)
    return nullptr;
  auto inserted = g_program_cache.emplace(key, std::move(program));
  return inserted.first->second.get();
}

static bool run_quantize_activation(
    ANERowwiseGEMMCache* const cache,
    PipelineValue<ANERowwiseTransformKernel>* const transform_pipeline,
    const CompiledProgram* const program,
    MTL::Buffer* const activation,
    const size_t activation_offset,
    MTL::Buffer* const weight,
    const size_t weight_offset,
    ccv_nnc_stream_context_t* const stream_context,
    std::string* const error_out)
{
  auto* const kernel = transform_pipeline->kernel;
  const MTL::Size activation_scale_grid_size = kernel->activationScaleGridSize(program->M);
  const MTL::Size activation_scale_threadgroup_size = kernel->activationScaleThreadgroupSize();
  const MTL::Size activation_quantize_grid_size = kernel->activationQuantizeGridSize(program->M, program->K);
  const MTL::Size activation_quantize_threadgroup_size = kernel->activationQuantizeThreadgroupSize();
  const auto append_weight_blit = [&](mtl_command_batch_t* const command_batch) -> bool {
    if (!command_batch || !command_batch->commandBuffer || !command_batch->commandEncoder || !weight || !cache->scratch.weight_surface_buffer) {
      if (error_out)
        *error_out = "failed to append weight upload to quantize command batch";
      return false;
    }
    command_batch->commandEncoder->endEncoding();
    command_batch->commandEncoder = nullptr;
    id<MTLBlitCommandEncoder> const blit_encoder = [bridge_command_buffer(command_batch->commandBuffer) blitCommandEncoder];
    if (!blit_encoder) {
      if (error_out)
        *error_out = "failed to create blit encoder for weight upload";
      return false;
    }
    [blit_encoder copyFromBuffer:bridge_buffer(weight)
                    sourceOffset:weight_offset
                        toBuffer:bridge_buffer(cache->scratch.weight_surface_buffer)
               destinationOffset:0
                            size:(size_t)program->N * program->K * sizeof(int8_t)];
    [blit_encoder endEncoding];
    return true;
  };

  mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(transform_pipeline->pipeline.get());
  encoder->useResource(activation, MTL::ResourceUsageRead);
  encoder->useResource(cache->scratch.activation_scales, MTL::ResourceUsageWrite);
  encoder->setBuffer(activation, activation_offset, 0);
  encoder->setBuffer(cache->scratch.activation_scales, 0, 1);
  encoder->dispatchThreadgroups(activation_scale_grid_size, activation_scale_threadgroup_size);
  command_batch->finishCommand(encoder);

  encoder = command_batch->startCommand();
  encoder->setComputePipelineState(transform_pipeline->second.get());
  encoder->useResource(activation, MTL::ResourceUsageRead);
  encoder->useResource(cache->scratch.activation_scales, MTL::ResourceUsageRead);
  encoder->useResource(cache->scratch.activation_surface_buffer, MTL::ResourceUsageWrite);
  encoder->setBuffer(activation, activation_offset, 0);
  encoder->setBuffer(cache->scratch.activation_scales, 0, 1);
  encoder->setBuffer(cache->scratch.activation_surface_buffer, 0, 2);
  encoder->dispatchThreadgroups(activation_quantize_grid_size, activation_quantize_threadgroup_size);
  command_batch->finishCommand(encoder);
  if (!append_weight_blit(command_batch))
    return false;
  if (stream_context &&
      (uint64_t)program->M * program->K <= kPrivateQuantCommitActivationElementsThreshold) {
    // Quant is a strict prerequisite for the synchronous ANE evaluate below.
    // Finish encoding onto the shared queue, but commit and wait on this one
    // command buffer directly so we don't also block on the stream's watermark
    // bookkeeping for older unrelated Metal work. This helps medium shapes more
    // than large ones; for very large quant launches, the normal stream path
    // sustains peak throughput better.
    id const mps_command_buffer =
      ccv_nnc_stream_context_finish_command_batch_encoding_and_return_mps_command_buffer(stream_context, command_batch);
    id<MTLCommandBuffer> const command_buffer =
      [((id<MTLCommandBuffer> (*)(id, SEL))objc_msgSend)(mps_command_buffer, @selector(commandBuffer)) retain];
    [mps_command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      [command_buffer release];
      if (error_out)
        *error_out = "activation quantize command failed";
      return false;
    }
    [command_buffer release];
    return true;
  }
  id<MTLCommandBuffer> const command_buffer = [bridge_command_buffer(command_batch->commandBuffer) retain];
  ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
  if (stream_context || command_buffer.status != MTLCommandBufferStatusCompleted)
    [command_buffer waitUntilCompleted];
  if (command_buffer.status != MTLCommandBufferStatusCompleted) {
    [command_buffer release];
    if (error_out)
      *error_out = "activation quantize command failed";
      return false;
  }
  [command_buffer release];
  return true;
}

static bool evaluate_program(
    ANERowwiseGEMMCache* const cache,
    const CompiledProgram* const program,
    std::string* const error_out)
{
  @autoreleasepool {
    struct ScopedPixelBuffers {
      CVPixelBufferRef activation = nullptr;
      CVPixelBufferRef weight = nullptr;
      CVPixelBufferRef output = nullptr;
      ~ScopedPixelBuffers()
      {
        if (activation)
          CFRelease(activation);
        if (weight)
          CFRelease(weight);
        if (output)
          CFRelease(output);
      }
    } pixel_buffers;
    MLModel* const model = (__bridge MLModel*)program->model;
    NSString* const x_name = (__bridge NSString*)program->x_name;
    NSString* const w_name = (__bridge NSString*)program->w_name;
    NSString* const y_name = (__bridge NSString*)program->y_name;
    if (!model || !cache->scratch.activation_surface || !cache->scratch.weight_surface || !cache->scratch.output_surface || !x_name || !w_name || !y_name) {
      if (error_out)
        *error_out = "CoreML rowwise program is missing runtime objects";
      return false;
    }
    NSArray<NSNumber*>* const activation_shape = make_shape(1, 1, program->K, cache->scratch.M);
    NSArray<NSNumber*>* const weight_shape = make_shape(1, 1, program->N, program->K);
    NSArray<NSNumber*>* const output_shape = make_shape(1, 1, program->N, cache->scratch.M);
    pixel_buffers.activation = create_pixel_buffer_for_surface(cache->scratch.activation_surface, activation_shape, MLMultiArrayDataTypeInt8, error_out);
    pixel_buffers.weight = create_pixel_buffer_for_surface(cache->scratch.weight_surface, weight_shape, MLMultiArrayDataTypeInt8, error_out);
    pixel_buffers.output = create_pixel_buffer_for_surface(cache->scratch.output_surface, output_shape, MLMultiArrayDataTypeFloat16, error_out);
    MLMultiArray* const activation_array = pixel_buffers.activation ? create_multiarray_with_pixel_buffer(pixel_buffers.activation, activation_shape, error_out) : nil;
    MLMultiArray* const weight_array = pixel_buffers.weight ? create_multiarray_with_pixel_buffer(pixel_buffers.weight, weight_shape, error_out) : nil;
    MLMultiArray* const output_array = pixel_buffers.output ? create_multiarray_with_pixel_buffer(pixel_buffers.output, output_shape, error_out) : nil;
    if (!activation_array || !weight_array || !output_array) {
      if (error_out && error_out->empty())
        *error_out = "failed to create CoreML multiarray wrappers from IOSurface";
      return false;
    }
    NSError* error = nil;
    NSDictionary* const inputs = @{
      x_name : [MLFeatureValue featureValueWithMultiArray:activation_array],
      w_name : [MLFeatureValue featureValueWithMultiArray:weight_array],
    };
    MLDictionaryFeatureProvider* const input_provider =
      [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:&error];
    if (!input_provider) {
      if (error_out)
        *error_out = "failed to create CoreML input provider: " + describe_nserror(error);
      return false;
    }
    MLPredictionOptions* const opts = [[MLPredictionOptions alloc] init];
    if (output_backing_key_supported()) {
      @try {
        [opts setOutputBackings:@{y_name : output_array}];
      } @catch (NSException* exception) {
        if (error_out)
          *error_out = exception.reason ? std::string(exception.reason.UTF8String) : "setOutputBackings failed";
        return false;
      }
    }
    id<MLFeatureProvider> const output_provider = [model predictionFromFeatures:input_provider options:opts error:&error];
    [input_provider release];
    [opts release];
    if (!output_provider) {
      if (error_out)
        *error_out = "CoreML evaluate failed: " + describe_nserror(error);
      return false;
    }
    MLMultiArray* result_array = [output_provider featureValueForName:y_name].multiArrayValue;
    if (!result_array) {
      for (NSString* feature_name in output_provider.featureNames) {
        result_array = [output_provider featureValueForName:feature_name].multiArrayValue;
        if (result_array)
          break;
      }
    }
    if (!result_array || !result_array.dataPointer) {
      if (error_out)
        *error_out = "CoreML output provider does not contain a readable multiarray";
      return false;
    }
    const size_t result_bytes = (size_t)result_array.count * bytes_per_ml_datatype(result_array.dataType);
    if (result_bytes != cache->scratch.output_surface_bytes) {
      if (error_out)
        *error_out = "CoreML output size does not match rowwise scratch";
      return false;
    }
    return true;
  }
}

static bool run_dequantize_output(
    ANERowwiseGEMMCache* const cache,
    PipelineValue<ANERowwiseTransformKernel>* const transform_pipeline,
    const CompiledProgram* const program,
    MTL::Buffer* const weight_buffer,
    const size_t weight_scale_offset,
    MTL::Buffer* const bias_buffer,
    const size_t bias_offset,
    MTL::Buffer* const output,
    const size_t output_offset,
    const uint32_t fused_bias,
    ccv_nnc_stream_context_t* const stream_context,
    std::string* const error_out)
{
  mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
  auto encoder = command_batch->startCommand();
  auto* const kernel = transform_pipeline->kernel;
  MTL::Buffer* const coreml_output_buffer = cache->scratch.output_surface_buffer;
  if (!coreml_output_buffer) {
    if (error_out)
      *error_out = "CoreML output buffer is not available for dequantize";
    return false;
  }
  encoder->setComputePipelineState(fused_bias ? transform_pipeline->fourth.get() : transform_pipeline->third.get());
  encoder->useResource(coreml_output_buffer, MTL::ResourceUsageRead);
  encoder->useResource(output, MTL::ResourceUsageWrite);
  encoder->useResource(cache->scratch.activation_scales, MTL::ResourceUsageRead);
  encoder->useResource(weight_buffer, MTL::ResourceUsageRead);
  encoder->setBuffer(coreml_output_buffer, 0, 0);
  encoder->setBuffer(output, output_offset, 1);
  encoder->setBuffer(cache->scratch.activation_scales, 0, 2);
  encoder->setBuffer(weight_buffer, weight_scale_offset, 3);
  if (fused_bias) {
    if (!bias_buffer) {
      if (error_out)
        *error_out = "bias buffer is not available for output bias add";
      return false;
    }
    encoder->useResource(bias_buffer, MTL::ResourceUsageRead);
    encoder->setBuffer(bias_buffer, bias_offset, 4);
  }
  encoder->dispatchThreadgroups(
      kernel->outputDequantizeGridSize(program->M, program->N),
      kernel->outputDequantizeThreadgroupSize());
  command_batch->finishCommand(encoder);
  id<MTLCommandBuffer> const command_buffer = stream_context ? nil : [bridge_command_buffer(command_batch->commandBuffer) retain];
  ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
  if (stream_context || (command_buffer && command_buffer.status != MTLCommandBufferStatusCompleted))
    [command_buffer waitUntilCompleted];
  if (command_buffer && command_buffer.status != MTLCommandBufferStatusCompleted) {
    [command_buffer release];
    if (error_out)
      *error_out = "output dequantize command failed";
    return false;
  }
  if (command_buffer)
    [command_buffer release];
  return true;
}

} // namespace

int ccv_nnc_mfa_run_ane_rowwise_gemm(
    ccv_nnc_mfa_context_t* const context,
    ccv_nnc_mfa_ane_rowwise_gemm_params_t params,
    mtl_buffer_t** tensors,
    size_t* tensor_offsets,
    ccv_nnc_stream_context_t* const stream_context)
{
  @autoreleasepool {
    CCV_NNC_MFA_PRECONDITION(context != nullptr);
    CCV_NNC_MFA_PRECONDITION(tensors != nullptr);
    CCV_NNC_MFA_PRECONDITION(tensor_offsets != nullptr);
    std::string error;
    ANERowwiseGEMMCache* const cache = get_or_create_cache(context, &error);
    if (!cache) {
      log_ane_rowwise_error(context, error);
      return 0;
    }
    MTL::Buffer* const activation = tensors[0];
    MTL::Buffer* const weight = tensors[1];
    MTL::Buffer* const output = tensors[2];
    MTL::Buffer* const bias = params.fused_bias ? tensors[3] : nullptr;
    PipelineValue<ANERowwiseTransformKernel>* const transform_pipeline = find_transform_pipeline(context, params);
    CompiledProgram* const program = find_or_create_program(params, &error);
    if (!program) {
      log_ane_rowwise_error(context, error);
      return 0;
    }
    if (!ensure_shared_scratch(cache, params, &error)) {
      log_ane_rowwise_error(context, error);
      return 0;
    }
    const size_t weight_scale_offset = tensor_offsets[1] + rowwise_8i_scale_offset(params.N, params.K);
    const size_t bias_offset = params.fused_bias ? tensor_offsets[3] : 0;
    if (!run_quantize_activation(cache, transform_pipeline, program, activation, tensor_offsets[0], weight, tensor_offsets[1], stream_context, &error)) {
      log_ane_rowwise_error(context, error);
      return 0;
    }
    if (!evaluate_program(cache, program, &error)) {
      log_ane_rowwise_error(context, error);
      return 0;
    }
    if (!run_dequantize_output(cache, transform_pipeline, program, weight, weight_scale_offset, bias, bias_offset, output, tensor_offsets[2], params.fused_bias, stream_context, &error)) {
      log_ane_rowwise_error(context, error);
      return 0;
    }
    return 1;
  }
}

void ccv_nnc_mfa_ane_rowwise_gemm_cleanup(ccv_nnc_mfa_context_t* const context)
{
  if (!context || !ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context))
    return;
  auto* const cache = (ANERowwiseGEMMCache*)ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(context);
  destroy_shared_scratch(&cache->scratch);
  destroy_surface_cache(&cache->activation_surface_cache);
  destroy_surface_cache(&cache->weight_surface_cache);
  destroy_surface_cache(&cache->output_surface_cache);
  delete cache;
  ccv_nnc_mfa_context_set_ane_rowwise_gemm_cache(context, nullptr);
}
