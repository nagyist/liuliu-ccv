#ifndef CastKernel_hpp
#define CastKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "CastDescriptor.hpp"

struct CastKernel {
  NS::SharedPtr<MTL::Library> library;
  
  std::string source;

  unsigned short threadgroupMemoryAllocation;

  /// The number of threads per group.
  MTL::Size threadgroupSize;

  uint8_t value;

  GEMMOperandPrecision fromMemoryPrecision;

  GEMMOperandPrecision memoryPrecision;

  CastKernel(CastKernelDescriptor descriptor, MTL::Device *const device);

private:
  unsigned short createThreadgroupMemoryAllocation() const noexcept;
  std::string createSource() const noexcept;
  std::string createConstants() const noexcept;
};

#endif /* CastKernel_hpp */

