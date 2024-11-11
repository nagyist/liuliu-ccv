#include "AddKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

AddKernel::AddKernel(AddKernelDescriptor descriptor, MTL::Device *const device) {

  value = descriptor.value;

  memoryPrecision = descriptor.memoryPrecision;

  source = createSource();

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();

  threadgroupSize = MTL::Size(256, 1, 1);

  // Compile the shader source.
  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

unsigned short AddKernel::createThreadgroupMemoryAllocation() const noexcept {
  return 0;
}

std::string AddKernel::createSource() const noexcept {
  std::string shader = createConstants() + "\n";
  if (value == 0) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  device real4 *src0 [[buffer(0)]],
  device real4 *src1 [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  destination[idx] = src0[idx] + src1[idx];
}
  )";
  } else if (value == 1) {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  device real4 *src0 [[buffer(0)]],
  device real4 *src1 [[buffer(1)]],
  device real4 *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = src0[idx] + src1[idx];
}
  )";
  } else {
    shader += R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = src0[idx] + src1[idx];
}
  )";
  }
  return shader;
}

std::string AddKernel::createConstants() const noexcept {

  std::string defines = "";
  if (value == 0 || value == 1) {
    if (memoryPrecision == GEMMOperandPrecision::FP32) {
      defines += std::string("typedef float4 real4;");
      defines += "\n";
    } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
      defines += std::string("typedef bfloat4 real4;");
      defines += "\n";
    } else {
      defines += std::string("typedef half4 real4;");
      defines += "\n";
    }
  } else {
    if (memoryPrecision == GEMMOperandPrecision::FP32) {
      defines += std::string("typedef float real;");
      defines += "\n";
    } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
      defines += std::string("typedef bfloat real;");
      defines += "\n";
    } else {
      defines += std::string("typedef half real;");
      defines += "\n";
    }
  }
  if (value != 0) {
    defines += "constant uint count [[function_constant(0)]];";
    defines += "\n";
  }
  return defines;
}
