#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "NAInt8SolAttentionKernel.hpp"
int main() {
 setbuf(stdout,nullptr);
 auto pool=NS::TransferPtr(NS::AutoreleasePool::alloc()->init());auto device=NS::TransferPtr(MTL::CreateSystemDefaultDevice());auto queue=NS::TransferPtr(device->newCommandQueue());auto ctx=ccv_nnc_init_mfa_context(device.get());
 for(auto s: {std::array<uint32_t,3>{2,257,3},{1,20480,2},{1,20481,56},{1,32769,2},{1,32768,56},{1,65536,56},{1,103982,56},{2,193,1},{1,257,7},{1,257,8},{1,257,15},{1,257,16},{1,257,31},{1,257,32},{1,20501,63},{1,20501,64}}) {
  const auto N=s[0],T=s[1],H=s[2];const size_t count=size_t(N)*T*H*128, bytes=count*2;
  auto in=NS::TransferPtr(device->newBuffer(bytes+512,MTL::ResourceStorageModeShared));auto out=NS::TransferPtr(device->newBuffer(size_t(N)*H*128*4+512,MTL::ResourceStorageModeShared));
  memset(out->contents(),0xa5,out->length());auto v=(_Float16*)((char*)in->contents()+258);auto m=(float*)((char*)out->contents()+256);
  std::mt19937 rng(42);std::uniform_real_distribution<float>r(-3,3);for(size_t i=0;i<count;++i)v[i]=r(rng)+(i%4==0?64:0);
  NAInt8SolAttentionDescriptor d={NAInt8SolAttentionStage::VMean,64,N,T,H,64};auto val=ctx->kernel_cache.findKernel<NAInt8SolAttentionKernel,NAInt8SolAttentionDescriptor,NAInt8SolAttentionKernelDescriptor>(d,device.get(),DeviceProperties());
  auto cb=queue->commandBuffer();auto enc=cb->computeCommandEncoder();enc->setComputePipelineState(val->pipeline.get());enc->setBuffer(in.get(),258,2);enc->setBuffer(out.get(),256,15);enc->dispatchThreadgroups(NAInt8SolAttentionKernel::vMeanThreadgroupsPerGrid(N,T,H),MTL::Size(NAInt8SolAttentionKernel::vMeanThreadgroupSize(T,H),1,1));enc->endEncoding();cb->commit();cb->waitUntilCompleted();if(cb->status()==MTL::CommandBufferStatusError)return 1;
  const uint32_t threads=T<=20480?256:128;size_t differences=0;double max_error=0;
  for(uint32_t n=0;n<N;++n)for(uint32_t h=0;h<H;++h)for(uint32_t d=0;d<128;++d){
   std::array<float,256> partial{};double exact=0;
   for(uint32_t i=0;i<threads;++i)for(uint32_t t=i;t<T;t+=threads){float x=v[((size_t(n)*T+t)*H+h)*128+d];partial[i]+=x;exact+=x;}
   for(uint32_t offset=16;offset;offset/=2){auto prev=partial;for(uint32_t i=0;i<threads;++i)partial[i]+=prev[(i/32)*32+((i%32)^offset)];}
   std::array<float,32> last{};for(uint32_t i=0;i<threads/32;++i)last[i]=partial[i*32];for(uint32_t offset=16;offset;offset/=2){auto prev=last;for(uint32_t i=0;i<32;++i)last[i]+=prev[i^offset];}
   const float expected=last[0]*(1.0f/float(T));const float actual=m[(n*H+h)*128+d];differences+=memcmp(&actual,&expected,4)!=0;if(!std::isfinite(actual))return 1;max_error=std::max(max_error,fabs(actual-exact/T));
  }
  for(size_t i=0;i<256;++i)if(((unsigned char*)out->contents())[i]!=0xa5||((unsigned char*)out->contents())[out->length()-256+i]!=0xa5)return 1;
  printf("N=%u T=%u H=%u tile=%u threads=%u shared=%zu cap=%zu differences=%zu max_double_error=%.9g\n",N,T,H,NAInt8SolAttentionKernel::vMeanVectorsPerTile(T,H),NAInt8SolAttentionKernel::vMeanThreadgroupSize(T,H),size_t(val->pipeline->staticThreadgroupMemoryLength()),size_t(device->maxThreadgroupMemoryLength()),differences,max_error);
  if(differences)return 1;
 }
 ccv_nnc_deinit_mfa_context(ctx);
}
