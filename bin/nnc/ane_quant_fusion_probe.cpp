// Compare the original two-dispatch ANE preparation with fused candidates.
// "selected" measures the fused shader directly (including the experimental
// wide-row variant, which is not enabled in production); the default sweeps
// experimental tile layouts. GPU timestamps exclude compilation and CPU setup.
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/ccv_nnc_mfa_ane_rowwise_internal.hpp"
#include "nnc/mfa/kernels/ANERowwiseTransformDescriptor.hpp"
#include "nnc/mfa/kernels/ANERowwiseTransformKernel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
extern "C" ccv_nnc_mfa_context_t* ccv_nnc_default_mfa_context(void);

static std::string candidate(int rows, int threads, bool retain, int k)
{
 std::string s = "\n#define ROWS " + std::to_string(rows) + "\n#define THREADS " + std::to_string(threads) + "\n#define RETAIN " + std::to_string(retain) + "\n#define SLOTS " + std::to_string((k + threads * 4 - 1) / (threads * 4)) + "\n";
 s += R"(
kernel void fused_quant(device const IO* src [[buffer(0)]], device IO* scales [[buffer(1)]], device char* dst [[buffer(2)]], uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint gid [[threadgroup_position_in_grid]]) {
 threadgroup float partial[ROWS][THREADS / 32];
 threadgroup char tile[ROWS][THREADS * 4];
 const uint row_in_tile = tid / THREADS;
 const uint row = gid * ROWS + row_in_tile;
 const uint local = tid % THREADS;
 const uint sg = local / 32;
 const uint base = source_offset(min(row, TOTAL_ROWS - 1));
 float maximum = 0;
 #if RETAIN
 float4 values[SLOTS];
 #endif
 for (uint g = 0; g < (K + THREADS * 4 - 1) / (THREADS * 4); ++g) {
  const uint col = g * THREADS * 4 + local * 4;
  float4 v = 0;
  for (uint c = 0; c < 4; ++c) if (col + c < K) v[c] = float(src[base + col + c]);
  #if RETAIN
  values[g] = v;
  #endif
  float4 a = abs(v);
  maximum = max(maximum, max(max(a.x, a.y), max(a.z, a.w)));
 }
 maximum = simd_max(maximum);
 if (lane == 0) partial[row_in_tile][sg] = maximum;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 maximum = 0;
 for (uint g = 0; g < THREADS / 32; ++g) maximum = max(maximum, partial[row_in_tile][g]);
 const IO stored_scale = IO(maximum > 0 ? maximum / 127.0f : 1.0f / 127.0f);
 if (local == 0) scales[row] = stored_scale;
 const float inv = float(stored_scale) > 0 ? 1.0f / float(stored_scale) : 127.0f;
 for (uint g = 0; g < (K + THREADS * 4 - 1) / (THREADS * 4); ++g) {
  const uint col = g * THREADS * 4 + local * 4;
  #if RETAIN
  float4 v = values[g];
  #else
  float4 v = 0;
  for (uint c = 0; c < 4; ++c) if (col + c < K) v[c] = float(src[base + col + c]);
  #endif
  const char4 q = char4(clamp(int4(rint(v * inv)), int4(-127), int4(127)));
  *reinterpret_cast<threadgroup char4*>(tile[row_in_tile] + local * 4) = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = tid; index < THREADS * 4 * (ROWS / 4); index += ROWS * THREADS) {
   const uint feature = index / (ROWS / 4);
   const uint r = (index % (ROWS / 4)) * 4;
   const uint output_k = g * THREADS * 4 + feature;
   if (output_k < K) {
    const char4 packed = char4(tile[r][feature], tile[r+1][feature], tile[r+2][feature], tile[r+3][feature]);
    reinterpret_cast<device char4*>(dst)[output_k * (PADDED_ROWS / 4) + gid * (ROWS / 4) + r / 4] = packed;
   }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
)";
 return s;
}
int main(int argc,char** argv) {
 if (argc < 3 || argc > 5) {
  fprintf(stderr, "usage: %s M K [type=16:FP16,32:FP32,121:BF16] [selected]\n", argv[0]);
  return 1;
 }
 const int m=atoi(argv[1]), k=atoi(argv[2]), type=argc>3?atoi(argv[3]):16;
 if (m <= 0 || k <= 0 || (type != 16 && type != 32 && type != 121) ||
     (argc > 4 && (strcmp(argv[4], "selected") || k > 16384))) return 1;
 auto pool=NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
 ccv_nnc_init(); auto ctx=ccv_nnc_default_mfa_context(); auto device=ccv_nnc_mfa_context_device(ctx);
 ANERowwiseTransformDescriptor d={}; d.M=m; d.paddedM=(m+127)/128*128; d.K=k; d.N=256; d.batchDimension=1;
 d.memoryPrecision=type==32?GEMMOperandPrecision::FP32:type==121?GEMMOperandPrecision::BF16:GEMMOperandPrecision::FP16;
 auto p=ccv_nnc_mfa_prepare_ane_rowwise_transform(ctx,d);
 auto constants=NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
 unsigned cv[]={d.M,d.paddedM,1,d.N,d.K,0,0,0,0,0,0};
 for(unsigned i=0;i<11;++i)constants->setConstantValue(cv+i,MTL::DataTypeUInt,i);
 NS::Error* error=nil;
 auto scale_function=NS::TransferPtr(p->kernel->library->newFunction(NS::String::string("compute_activation_scales",NS::UTF8StringEncoding),constants.get(),&error));
 auto baseline_scale=NS::TransferPtr(device->newComputePipelineState(scale_function.get(),&error));
 if(error)return 3;
 auto queue=NS::TransferPtr(device->newCommandQueue());
 auto src=NS::TransferPtr(device->newBuffer((size_t)m*k*4,MTL::ResourceStorageModeShared));
 auto scales=NS::TransferPtr(device->newBuffer(d.paddedM*4,MTL::ResourceStorageModeShared));
 auto out=NS::TransferPtr(device->newBuffer((size_t)d.paddedM*k,MTL::ResourceStorageModeShared));
 unsigned state=42; std::vector<float> input((size_t)m*k);
 for(size_t i=0;i<input.size();++i) {state=state*1664525u+1013904223u; input[i]=i<(size_t)k?0:((int)(state>>16)-32768)/32768.f;}
 if(type==32) memcpy(src->contents(),input.data(),input.size()*4); else if(type==121) ccv_float_to_bfloat(input.data(),(uint16_t*)src->contents(),input.size()); else ccv_float_to_half_precision(input.data(),(uint16_t*)src->contents(),input.size());
 auto encode=[&](MTL::CommandBuffer* cb, MTL::ComputePipelineState* fused, int rows, int threads) {
  auto e=cb->computeCommandEncoder(); e->setComputePipelineState(fused?fused:baseline_scale.get()); e->setBuffer(src.get(),0,0); e->setBuffer(scales.get(),0,1);
  if(fused) e->setBuffer(out.get(),0,2);
  e->dispatchThreadgroups(MTL::Size(fused?d.paddedM/rows:d.paddedM,1,1),MTL::Size(fused?rows*threads:256,1,1)); e->endEncoding();
  if(!fused) { e=cb->computeCommandEncoder(); e->setComputePipelineState(p->second.get()); e->setBuffer(src.get(),0,0);e->setBuffer(scales.get(),0,1);e->setBuffer(out.get(),0,2);e->dispatchThreadgroups(p->kernel->activationQuantizeGridSize(d.paddedM,k),p->kernel->activationQuantizeThreadgroupSize());e->endEncoding(); }
 };
 auto measure=[&](MTL::ComputePipelineState* fused,int rows,int threads,int iters) {
  auto cb=queue->commandBuffer(); for(int i=0;i<iters;++i)encode(cb,fused,rows,threads); cb->commit();cb->waitUntilCompleted();if(cb->status()!=MTL::CommandBufferStatusCompleted)exit(2);return (cb->GPUEndTime()-cb->GPUStartTime())*1000/iters;
 };
 measure(nullptr,0,0,1); std::vector<char> expected((size_t)d.paddedM*k);memcpy(expected.data(),out->contents(),expected.size());std::vector<char> expected_scales(d.paddedM*(type==32?4:2));memcpy(expected_scales.data(),scales->contents(),expected_scales.size());
 printf("M,K,type,rows,threads,retain,baseline_ms,fused_ms,change_pct,byte_errors,scale_errors\n");
 for(bool retain:{false,true})for(int rows:{4,8,16,32})for(int threads:{32,64,128,256}) {
  if(rows*threads>1024)continue;
  const bool selected = argc > 4;
  if(selected && (rows != (k <= 8192 ? 8 : 4) || threads != (k <= 8192 ? 128 : 256) || !retain))continue;
  std::string s=selected?p->kernel->source:p->kernel->source+"\n#define IO "+(type==32?"float":type==121?"bfloat":"half")+candidate(rows,threads,retain,k);
  NS::Error* err=nil;auto lib=NS::TransferPtr(device->newLibrary(NS::String::string(s.c_str(),NS::UTF8StringEncoding),nil,&err)); if(err){fprintf(stderr,"%s\n",err->localizedDescription()->utf8String());return 3;}
  auto c=NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init()); unsigned vals[]={d.M,d.paddedM,1,d.N,d.K,0,0,0,0,0,0};for(unsigned i=0;i<11;++i)c->setConstantValue(vals+i,MTL::DataTypeUInt,i);
  auto f=NS::TransferPtr(lib->newFunction(NS::String::string(selected?"quantize_activation":"fused_quant",NS::UTF8StringEncoding),c.get(),&err));auto fp=NS::TransferPtr(device->newComputePipelineState(f.get(),&err)); if(err){fprintf(stderr,"%s\n",err->localizedDescription()->utf8String());return 4;}
  if(fp->maxTotalThreadsPerThreadgroup()<(unsigned)(rows*threads))continue;
  measure(fp.get(),rows,threads,1);size_t errors=0;for(size_t i=0;i<expected.size();++i)errors+=expected[i]!=((char*)out->contents())[i];bool scale_errors=memcmp(expected_scales.data(),scales->contents(),expected_scales.size())!=0;
  std::vector<double>a,b;for(int i=0;i<(selected?31:9);++i){double x,y;if(i%2){y=measure(fp.get(),rows,threads,10);x=measure(nullptr,0,0,10);}else{x=measure(nullptr,0,0,10);y=measure(fp.get(),rows,threads,10);}a.push_back(x);b.push_back(y);if(selected)printf("# pair,%d,%.9f,%.9f\n",i,x,y);}std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
  printf("%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f,%zu,%d\n",m,k,type,rows,threads,retain,a[a.size()/2],b[b.size()/2],100*(b[b.size()/2]/a[a.size()/2]-1),errors,scale_errors);fflush(stdout);
 }
}
