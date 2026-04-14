#include "ccv_nnc_mfa_ane_rowwise_coreml.hpp"

#include "ccv_nnc_mfa_error.hpp"

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#if __has_include(<IOSurface/IOSurface.h>)
#import <IOSurface/IOSurface.h>
#else
#import <IOSurface/IOSurfaceRef.h>
#endif
#import <Metal/Metal.h>
#import <objc/message.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
mtl_command_batch_t* ccv_nnc_stream_context_start_command_batch(ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_stream_context_finish_command_batch(ccv_nnc_stream_context_t* const stream_context, mtl_command_batch_t* command_batch);
id ccv_nnc_stream_context_finish_command_batch_encoding_and_return_mps_command_buffer(
    ccv_nnc_stream_context_t* const stream_context,
    mtl_command_batch_t* command_batch);
}

namespace {

constexpr MTLResourceOptions kSharedResourceOptions = MTLResourceStorageModeShared;
constexpr size_t kANEActivationSurfaceCacheLimitBytes = 512 * 1024 * 1024;
constexpr size_t kANEWeightSurfaceCacheLimitBytes = 128 * 1024 * 1024;
constexpr size_t kANEOutputSurfaceCacheLimitBytes = 1024 * 1024 * 1024;
constexpr size_t kANEScratchBufferAlignment = 16 * 1024;

static size_t align_up(const size_t value, const size_t alignment) noexcept
{
  return (value + alignment - 1) & ~(alignment - 1);
}

static void set_error(char* const error_out, const size_t error_out_size, const std::string& error)
{
  if (!error_out || error_out_size == 0)
    return;
  if (error.empty()) {
    error_out[0] = '\0';
    return;
  }
  const size_t copy_size = std::min(error_out_size - 1, error.size());
  memcpy(error_out, error.data(), copy_size);
  error_out[copy_size] = '\0';
}

static inline id<MTLCommandBuffer> bridge_command_buffer(mtl_command_buffer_t* const command_buffer)
{
  return (__bridge id<MTLCommandBuffer>)(void*)command_buffer;
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

static mtl_buffer_t* create_surface_buffer(mtl_device_t* const device, IOSurfaceRef const surface, const size_t bytes)
{
  if (!surface)
    return nullptr;
  void* const base_address = IOSurfaceGetBaseAddress(surface);
  return device->newBuffer(base_address, bytes, kSharedResourceOptions, nil);
}

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

static std::unordered_map<ProgramKey, std::unique_ptr<CompiledProgram>, ProgramKeyHash> g_program_cache;

struct CachedSurface {
  size_t width;
  size_t height;
  size_t alloc_bytes;
  CVPixelBufferRef pixel_buffer;
  IOSurfaceRef surface;
  mtl_buffer_t* buffer;
  uint64_t last_used;
};

struct SurfaceCache {
  std::vector<std::unique_ptr<CachedSurface>> entries;
  size_t total_alloc_bytes = 0;
  size_t limit_bytes = 0;
  uint64_t lru_clock = 0;
};

struct SharedScratch {
  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t K = 0;
  size_t activation_surface_bytes = 0;
  size_t weight_surface_bytes = 0;
  size_t output_surface_bytes = 0;
  size_t activation_scale_bytes = 0;
  size_t activation_scale_capacity_bytes = 0;
  CVPixelBufferRef activation_pixel_buffer = nullptr;
  CVPixelBufferRef weight_pixel_buffer = nullptr;
  CVPixelBufferRef output_pixel_buffer = nullptr;
  IOSurfaceRef activation_surface = nullptr;
  IOSurfaceRef weight_surface = nullptr;
  IOSurfaceRef output_surface = nullptr;
  mtl_buffer_t* activation_surface_buffer = nullptr;
  mtl_buffer_t* weight_surface_buffer = nullptr;
  mtl_buffer_t* output_surface_buffer = nullptr;
  mtl_buffer_t* activation_scales = nullptr;
};

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
  scratch->activation_pixel_buffer = nullptr;
  scratch->weight_pixel_buffer = nullptr;
  scratch->output_pixel_buffer = nullptr;
  scratch->activation_surface = nullptr;
  scratch->weight_surface = nullptr;
  scratch->output_surface = nullptr;
  scratch->activation_surface_buffer = nullptr;
  scratch->weight_surface_buffer = nullptr;
  scratch->output_surface_buffer = nullptr;
  scratch->activation_scales = nullptr;
}

static void evict_surface_cache_entries(SurfaceCache* const cache, CachedSurface* const keep_entry)
{
  while (cache->limit_bytes > 0 && cache->total_alloc_bytes > cache->limit_bytes) {
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
    mtl_device_t* const device,
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
  return [[[NSMutableDictionary alloc] initWithDictionary:@{
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
  }] autorelease];
}

static NSString* build_metadata_json(const uint32_t K, const uint32_t N, const uint32_t padded_M)
{
  NSMutableDictionary* const root = default_metadata_root();
  NSArray<NSNumber*>* const x_shape = make_shape(1, 1, K, padded_M);
  NSArray<NSNumber*>* const w_shape = make_shape(1, 1, N, K);
  NSArray<NSNumber*>* const y_shape = make_shape(1, 1, N, padded_M);
  root[@"inputSchema"] = @[
    schema_entry_for_name(@"x", "Int8", x_shape),
    schema_entry_for_name(@"w", "Int8", w_shape),
  ];
  root[@"outputSchema"] = @[
    schema_entry_for_name(@"mm", "Float16", y_shape),
  ];
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

static std::unique_ptr<CompiledProgram> compile_program(
    const uint32_t padded_M,
    const uint32_t N,
    const uint32_t K,
    std::string* const error_out)
{
  @autoreleasepool {
    const float model_scale = 1.0f / std::sqrt((float)K);
    NSString* const temp_directory = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.mlmodelc", [[NSUUID UUID] UUIDString]]];
    NSFileManager* const file_manager = [NSFileManager defaultManager];
    NSError* error = nil;
    if (![file_manager createDirectoryAtPath:[temp_directory stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:&error]) {
      if (error_out)
        *error_out = "failed to create manual .mlmodelc: " + describe_nserror(error);
      return {};
    }
    NSData* const coremldata = synthesized_coremldata(K, N, padded_M);
    if (![coremldata writeToFile:[temp_directory stringByAppendingPathComponent:@"coremldata.bin"] atomically:YES]) {
      if (error_out)
        *error_out = "failed to write coremldata.bin";
      return {};
    }
    if (!write_text_file(generate_dynamic_i8_two_input_matmul_tx_scaled_mil(K, N, padded_M, model_scale),
                         [temp_directory stringByAppendingPathComponent:@"model.mil"], error_out))
      return {};
    if (!write_text_file(build_metadata_json(K, N, padded_M),
                         [temp_directory stringByAppendingPathComponent:@"metadata.json"], error_out))
      return {};
    if (!write_text_file(build_manifest_json(@"model.mil", @"weights"),
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
    return std::unique_ptr<CompiledProgram>(new CompiledProgram {
      padded_M,
      N,
      K,
      CFBridgingRetain(model),
      CFBridgingRetain(x_name),
      CFBridgingRetain(w_name),
      CFBridgingRetain(y_name),
    });
  }
}

} // namespace

struct ccv_nnc_mfa_ane_rowwise_coreml_cache_s {
  mtl_device_t* device = nullptr;
  SharedScratch scratch;
  SurfaceCache activation_surface_cache;
  SurfaceCache weight_surface_cache;
  SurfaceCache output_surface_cache;
};

struct ccv_nnc_mfa_ane_rowwise_coreml_program_s {
  CompiledProgram* program;
};

extern "C" {

ccv_nnc_mfa_ane_rowwise_coreml_cache_t* ccv_nnc_mfa_ane_rowwise_coreml_cache_create(mtl_device_t* device)
{
  auto* const cache = new ccv_nnc_mfa_ane_rowwise_coreml_cache_t();
  cache->device = device;
  cache->activation_surface_cache.limit_bytes = kANEActivationSurfaceCacheLimitBytes;
  cache->weight_surface_cache.limit_bytes = kANEWeightSurfaceCacheLimitBytes;
  cache->output_surface_cache.limit_bytes = kANEOutputSurfaceCacheLimitBytes;
  return cache;
}

void ccv_nnc_mfa_ane_rowwise_coreml_cache_destroy(ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache)
{
  if (!cache)
    return;
  destroy_shared_scratch(&cache->scratch);
  destroy_surface_cache(&cache->activation_surface_cache);
  destroy_surface_cache(&cache->weight_surface_cache);
  destroy_surface_cache(&cache->output_surface_cache);
  delete cache;
}

int ccv_nnc_mfa_ane_rowwise_coreml_cache_ensure_scratch(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    uint32_t padded_M,
    uint32_t N,
    uint32_t K,
    char* error_out,
    size_t error_out_size)
{
  std::string error;
  if (!cache || !cache->device) {
    set_error(error_out, error_out_size, "ANE rowwise cache is not initialized");
    return 0;
  }
  const size_t activation_surface_bytes = (size_t)K * padded_M * sizeof(int8_t);
  const size_t weight_surface_bytes = (size_t)N * K * sizeof(int8_t);
  const size_t output_surface_bytes = (size_t)N * padded_M * sizeof(uint16_t);
  const size_t activation_scale_bytes = (size_t)padded_M * sizeof(uint16_t);
  SharedScratch& scratch = cache->scratch;
  const bool scratch_shape_matches = (scratch.M == padded_M && scratch.N == N && scratch.K == K);
  if (scratch_shape_matches &&
      scratch.activation_surface && scratch.weight_surface && scratch.output_surface &&
      scratch.activation_surface_buffer && scratch.weight_surface_buffer &&
      scratch.output_surface_buffer && scratch.activation_scales &&
      scratch.activation_scale_capacity_bytes >= activation_scale_bytes) {
    set_error(error_out, error_out_size, "");
    return 1;
  }
  CachedSurface* const activation_surface_entry =
      find_or_create_cached_surface(&cache->activation_surface_cache, cache->device, padded_M, K, MLMultiArrayDataTypeInt8, &error);
  CachedSurface* const weight_surface_entry =
      find_or_create_cached_surface(&cache->weight_surface_cache, cache->device, K, N, MLMultiArrayDataTypeInt8, &error);
  CachedSurface* const output_surface_entry =
      find_or_create_cached_surface(&cache->output_surface_cache, cache->device, padded_M, N, MLMultiArrayDataTypeFloat16, &error);
  scratch.activation_surface = activation_surface_entry ? activation_surface_entry->surface : nullptr;
  scratch.weight_surface = weight_surface_entry ? weight_surface_entry->surface : nullptr;
  scratch.output_surface = output_surface_entry ? output_surface_entry->surface : nullptr;
  scratch.activation_pixel_buffer = activation_surface_entry ? activation_surface_entry->pixel_buffer : nullptr;
  scratch.weight_pixel_buffer = weight_surface_entry ? weight_surface_entry->pixel_buffer : nullptr;
  scratch.output_pixel_buffer = output_surface_entry ? output_surface_entry->pixel_buffer : nullptr;
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
  scratch.N = N;
  scratch.K = K;
  scratch.activation_surface_bytes = activation_surface_bytes;
  scratch.weight_surface_bytes = weight_surface_bytes;
  scratch.output_surface_bytes = output_surface_bytes;
  scratch.activation_scale_bytes = activation_scale_bytes;
  if (!scratch.activation_surface || !scratch.weight_surface || !scratch.output_surface ||
      !scratch.activation_surface_buffer || !scratch.weight_surface_buffer ||
      !scratch.output_surface_buffer || !scratch.activation_scales) {
    if (error.empty())
      error = "failed to allocate CoreML rowwise scratch";
    destroy_surface_cache(&cache->activation_surface_cache);
    destroy_surface_cache(&cache->weight_surface_cache);
    destroy_surface_cache(&cache->output_surface_cache);
    destroy_shared_scratch(&scratch);
    set_error(error_out, error_out_size, error);
    return 0;
  }
  set_error(error_out, error_out_size, "");
  return 1;
}

mtl_buffer_t* ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_surface_buffer(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache)
{
  return cache ? cache->scratch.activation_surface_buffer : nullptr;
}

mtl_buffer_t* ccv_nnc_mfa_ane_rowwise_coreml_cache_weight_surface_buffer(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache)
{
  return cache ? cache->scratch.weight_surface_buffer : nullptr;
}

mtl_buffer_t* ccv_nnc_mfa_ane_rowwise_coreml_cache_output_surface_buffer(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache)
{
  return cache ? cache->scratch.output_surface_buffer : nullptr;
}

mtl_buffer_t* ccv_nnc_mfa_ane_rowwise_coreml_cache_activation_scales_buffer(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache)
{
  return cache ? cache->scratch.activation_scales : nullptr;
}

ccv_nnc_mfa_ane_rowwise_coreml_program_t* ccv_nnc_mfa_ane_rowwise_coreml_find_or_create_program(
    uint32_t padded_M,
    uint32_t N,
    uint32_t K,
    char* error_out,
    size_t error_out_size)
{
  const ProgramKey key = {
    .M = padded_M,
    .N = N,
    .K = K,
  };
  auto it = g_program_cache.find(key);
  if (it == g_program_cache.end()) {
    std::string error;
    std::unique_ptr<CompiledProgram> program = compile_program(padded_M, N, K, &error);
    if (!program) {
      set_error(error_out, error_out_size, error);
      return nullptr;
    }
    it = g_program_cache.emplace(key, std::move(program)).first;
  }
  auto* const handle = new ccv_nnc_mfa_ane_rowwise_coreml_program_t();
  handle->program = it->second.get();
  set_error(error_out, error_out_size, "");
  return handle;
}

void ccv_nnc_mfa_ane_rowwise_coreml_program_release(
    ccv_nnc_mfa_ane_rowwise_coreml_program_t* program)
{
  delete program;
}

uint32_t ccv_nnc_mfa_ane_rowwise_coreml_program_M(
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* program)
{
  return program && program->program ? program->program->M : 0;
}

uint32_t ccv_nnc_mfa_ane_rowwise_coreml_program_N(
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* program)
{
  return program && program->program ? program->program->N : 0;
}

uint32_t ccv_nnc_mfa_ane_rowwise_coreml_program_K(
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* program)
{
  return program && program->program ? program->program->K : 0;
}

int ccv_nnc_mfa_ane_rowwise_coreml_evaluate(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    const ccv_nnc_mfa_ane_rowwise_coreml_program_t* program_handle,
    char* error_out,
    size_t error_out_size)
{
  std::string error;
  @autoreleasepool {
    CompiledProgram* const program = program_handle ? program_handle->program : nullptr;
    MLModel* const model = program ? (__bridge MLModel*)program->model : nil;
    NSString* const x_name = program ? (__bridge NSString*)program->x_name : nil;
    NSString* const w_name = program ? (__bridge NSString*)program->w_name : nil;
    NSString* const y_name = program ? (__bridge NSString*)program->y_name : nil;
    if (!cache || !program || !model || !cache->scratch.activation_surface || !cache->scratch.weight_surface ||
        !cache->scratch.output_surface || !x_name || !w_name || !y_name) {
      error = "CoreML rowwise program is missing runtime objects";
      set_error(error_out, error_out_size, error);
      return 0;
    }
    NSArray<NSNumber*>* const activation_shape = make_shape(1, 1, program->K, cache->scratch.M);
    NSArray<NSNumber*>* const weight_shape = make_shape(1, 1, program->N, program->K);
    NSArray<NSNumber*>* const output_shape = make_shape(1, 1, program->N, cache->scratch.M);
    MLMultiArray* const activation_array =
        cache->scratch.activation_pixel_buffer ? create_multiarray_with_pixel_buffer(cache->scratch.activation_pixel_buffer, activation_shape, &error) : nil;
    MLMultiArray* const weight_array =
        cache->scratch.weight_pixel_buffer ? create_multiarray_with_pixel_buffer(cache->scratch.weight_pixel_buffer, weight_shape, &error) : nil;
    MLMultiArray* const output_array =
        cache->scratch.output_pixel_buffer ? create_multiarray_with_pixel_buffer(cache->scratch.output_pixel_buffer, output_shape, &error) : nil;
    if (!activation_array || !weight_array || !output_array) {
      if (error.empty())
        error = "failed to create CoreML multiarray wrappers from IOSurface";
      set_error(error_out, error_out_size, error);
      return 0;
    }
    NSError* ns_error = nil;
    NSDictionary* const inputs = @{
      x_name : [MLFeatureValue featureValueWithMultiArray:activation_array],
      w_name : [MLFeatureValue featureValueWithMultiArray:weight_array],
    };
    MLDictionaryFeatureProvider* const input_provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:&ns_error];
    if (!input_provider) {
      error = "failed to create CoreML input provider: " + describe_nserror(ns_error);
      set_error(error_out, error_out_size, error);
      return 0;
    }
    MLPredictionOptions* const opts = [[MLPredictionOptions alloc] init];
    if (output_backing_key_supported()) {
      @try {
        [opts setOutputBackings:@{y_name : output_array}];
      } @catch (NSException* exception) {
        [input_provider release];
        [opts release];
        error = exception.reason ? std::string(exception.reason.UTF8String) : "setOutputBackings failed";
        set_error(error_out, error_out_size, error);
        return 0;
      }
    }
    id<MLFeatureProvider> const output_provider = [model predictionFromFeatures:input_provider options:opts error:&ns_error];
    [input_provider release];
    [opts release];
    if (!output_provider) {
      error = "CoreML evaluate failed: " + describe_nserror(ns_error);
      set_error(error_out, error_out_size, error);
      return 0;
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
      error = "CoreML output provider does not contain a readable multiarray";
      set_error(error_out, error_out_size, error);
      return 0;
    }
    const size_t result_bytes = (size_t)result_array.count * bytes_per_ml_datatype(result_array.dataType);
    if (result_bytes != cache->scratch.output_surface_bytes) {
      error = "CoreML output size does not match rowwise scratch";
      set_error(error_out, error_out_size, error);
      return 0;
    }
  }
  set_error(error_out, error_out_size, "");
  return 1;
}

int ccv_nnc_mfa_ane_rowwise_coreml_append_weight_upload(
    ccv_nnc_mfa_ane_rowwise_coreml_cache_t* cache,
    mtl_command_batch_t* command_batch,
    mtl_buffer_t* weight,
    size_t weight_offset,
    size_t weight_bytes,
    char* error_out,
    size_t error_out_size)
{
  if (!cache || !command_batch || !command_batch->commandBuffer || !command_batch->commandEncoder || !weight ||
      !cache->scratch.weight_surface_buffer) {
    set_error(error_out, error_out_size, "failed to append weight upload to quantize command batch");
    return 0;
  }
  command_batch->commandEncoder->endEncoding();
  command_batch->commandEncoder = nullptr;
  id<MTLBlitCommandEncoder> const blit_encoder =
      [bridge_command_buffer(command_batch->commandBuffer) blitCommandEncoder];
  if (!blit_encoder) {
    set_error(error_out, error_out_size, "failed to create blit encoder for weight upload");
    return 0;
  }
  [blit_encoder copyFromBuffer:(__bridge id<MTLBuffer>)(void*)weight
                  sourceOffset:weight_offset
                      toBuffer:(__bridge id<MTLBuffer>)(void*)cache->scratch.weight_surface_buffer
             destinationOffset:0
                          size:weight_bytes];
  [blit_encoder endEncoding];
  return 1;
}

int ccv_nnc_mfa_ane_rowwise_finish_command_batch_and_wait(
    ccv_nnc_stream_context_t* stream_context,
    mtl_command_batch_t* command_batch,
    int use_mps_wrapper,
    char* error_out,
    size_t error_out_size)
{
  if (!command_batch) {
    set_error(error_out, error_out_size, "command batch is not available");
    return 0;
  }
  if (use_mps_wrapper) {
    id const mps_command_buffer =
        ccv_nnc_stream_context_finish_command_batch_encoding_and_return_mps_command_buffer(stream_context, command_batch);
    id<MTLCommandBuffer> const command_buffer =
        [((id<MTLCommandBuffer> (*)(id, SEL))objc_msgSend)(mps_command_buffer, @selector(commandBuffer)) retain];
    [mps_command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      [command_buffer release];
      set_error(error_out, error_out_size, "command buffer execution failed");
      return 0;
    }
    [command_buffer release];
    set_error(error_out, error_out_size, "");
    return 1;
  }
  id<MTLCommandBuffer> const command_buffer = [bridge_command_buffer(command_batch->commandBuffer) retain];
  ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
  if (stream_context || (command_buffer && command_buffer.status != MTLCommandBufferStatusCompleted))
    [command_buffer waitUntilCompleted];
  if (command_buffer && command_buffer.status != MTLCommandBufferStatusCompleted) {
    [command_buffer release];
    set_error(error_out, error_out_size, "command buffer execution failed");
    return 0;
  }
  if (command_buffer)
    [command_buffer release];
  set_error(error_out, error_out_size, "");
  return 1;
}

} // extern "C"
