#ifndef GUARD_ccv_nnc_mfa_gelu_hpp
#define GUARD_ccv_nnc_mfa_gelu_hpp

typedef struct {
  uint8_t gradient;
  uint8_t tanh;
  uint64_t data_type;
  uint32_t length;
} ccv_nnc_mfa_gelu_params_t;

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

namespace ccv {
namespace nnc {
namespace mfa {
namespace gelu {

class hash {
public:
  uint64_t data_type;
  uint32_t astride[3];
  uint32_t bstride[3];
  uint32_t cstride[3];
  uint32_t dim[4];

  hash(ccv_nnc_mfa_gelu_params_t);
};

class pipeline {
public:
  NS::SharedPtr<MTL::ComputePipelineState> gelu_pso;
  
  MTL::Size grid_size;
  MTL::Size group_size;
  
  pipeline(context* context, hash hash);
};

} // namespace gelu
} // namespace mfa
} // namespace nnc
} // namespace ccv

extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_gelu(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gelu_params_t params);
void ccv_nnc_mfa_encode_gelu(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gelu_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
