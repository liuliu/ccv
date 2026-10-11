// Compare tip and current source generators without compiling Metal libraries.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include "SnapshotNAInt8AttentionDescriptor.hpp"
#include "SnapshotNAInt8AttentionKernel.hpp"
#include "TipNAInt8AttentionDescriptor.hpp"
#include "TipNAInt8AttentionKernel.hpp"
template <typename T> T descriptor(unsigned r, unsigned c, unsigned d, unsigned b,
    unsigned hq, unsigned hk, unsigned causal, unsigned dynamic, unsigned precision,
    unsigned mode, unsigned hadamard, unsigned masked) {
  T x;
  x.matrixDimensions = simd::uint3{r,c,d}; x.batchDimension=b; x.Hq=hq; x.Hk=hk;
  x.scale=1/std::sqrt(float(d)); x.isCausal=causal; x.loadR=dynamic&1; x.loadC=dynamic&2;
  x.type=mode==0 ? AttentionKernelType::forward : (mode==1 ? AttentionKernelType::backwardQuery : AttentionKernelType::backwardKeyValue);
  x.ioPrecision=precision==0 ? GEMMOperandPrecision::FP16 : (precision==1 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP32);
  x.lowPrecisionIntermediates=precision!=2; x.qkHadamard=hadamard; x.masked=masked;
  x.batchStrides[AttentionOperand::Q]=b>1 ? r*hq*d : 0;
  x.batchStrides[AttentionOperand::K]=b>1 ? c*hk*d : 0;
  x.batchStrides[AttentionOperand::V]=b>1 ? c*hk*d : 0;
  x.batchStrides[AttentionOperand::O]=b>1 ? r*hq*d : 0;
  return x;
}
int main(int argc, char** argv) {
  std::ifstream input(argv[1]);
  unsigned r,c,d,b,hq,hk,causal,dynamic,count=0,bounds_only=0;
  while (input >> r >> c >> d >> b >> hq >> hk >> causal >> dynamic)
    for (unsigned core_count : {10u,40u})
    for (unsigned precision=0; precision<3; ++precision)
    for (unsigned mode=0; mode<3; ++mode)
    for (unsigned hadamard=0; hadamard<(mode==0?2u:1u); ++hadamard)
    for (unsigned masked=0; masked<2; ++masked) {
      auto a=descriptor<SnapshotNAInt8AttentionDescriptor>(r,c,d,b,hq,hk,causal,dynamic,precision,mode,hadamard,masked);
      auto z=descriptor<TipNAInt8AttentionDescriptor>(r,c,d,b,hq,hk,causal,dynamic,precision,mode,hadamard,masked);
      if (!a.quantizePV) return 2;
      const DeviceProperties properties{core_count};
      SnapshotNAInt8AttentionKernel ak(SnapshotNAInt8AttentionKernelDescriptor(a,properties),nullptr);
      TipNAInt8AttentionKernel zk(TipNAInt8AttentionKernelDescriptor(z,properties),nullptr);
      const auto ag=ak.threadgroupsPerGrid(b,r), zg=zk.threadgroupsPerGrid(b,r);
      // The shared bounds fix also repairs backward quantizer entry points.
      // Compare against exactly those substitutions, not a whitespace-normalized
      // or otherwise weakened source comparison. Forward stays byte-identical.
      std::string expected = zk.source;
      if (mode != 0) {
        auto enableBounds = [&](const std::string& name) {
          const std::string from = name + ",\n      false, ";
          const auto offset = expected.find(from);
          if (offset != std::string::npos)
            expected.replace(offset, from.size(), name + ",\n      true, ");
        };
        if (zk.blockDimensions[0] % zk.qScaleTileSize != 0)
          enableBounds("QUANTIZE_Q_SEQUENCE");
        if (zk.blockDimensions[1] % zk.kvScaleTileSize != 0)
          enableBounds("QUANTIZE_KV_SEQUENCE");
      }
      bounds_only += expected != zk.source;
      if (ak.source!=expected || ak.threadgroupMemoryAllocation()!=zk.threadgroupMemoryAllocation() ||
          ag.width!=zg.width || ag.height!=zg.height || ag.depth!=zg.depth ||
          ak.executionSIMDGroups!=zk.executionSIMDGroups ||
          ak.vMeanPartialBytes(b,c)!=zk.vMeanPartialBytes(b,c)) {
        printf("mismatch R=%u C=%u D=%u cores=%u precision=%u mode=%u hadamard=%u masked=%u source_equal=%d\n",r,c,d,core_count,precision,mode,hadamard,masked,ak.source==zk.source);
        std::ofstream("/private/tmp/ccv-pv-current.metal")<<ak.source;
        std::ofstream("/private/tmp/ccv-pv-tip.metal")<<zk.source;
        return 1;
      }
      ++count;
    }
  printf("default_source_and_dispatch_identical=%u backward_quantizer_bounds_only=%u total=%u\n",count-bounds_only,bounds_only,count);
  return count ? 0 : 3;
}
