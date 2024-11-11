#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include "v2/AddDescriptor.hpp"
#include "v2/AddKernel.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_add(mfa::context* context, ccv_nnc_mfa_add_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_add(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_add_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 3);

  AddDescriptor descriptor;
  descriptor.memoryPrecision = (params.data_type == MTL::DataTypeFloat) ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::FP16;
  descriptor.length = params.length;

  if (params.length % (4 * 256) == 0) {
    descriptor.value = 0;
  } else if (params.length % 4 == 0) {
    descriptor.value = 1;
  } else {
    descriptor.value = 2;
  }

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->v2_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<AddKernel, AddDescriptor, AddKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  
  if (tensors[0] == tensors[2]) {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  } else if (tensors[1] == tensors[2]) {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  }

  unsigned int count;
  if (params.length % 4 == 0) {
    count = params.length / 4;
  } else {
    count = params.length;
  }
  const int num_blocks = (count + 255) / 256;
  MTL::Size gridSize = MTL::Size(num_blocks, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.depth > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
