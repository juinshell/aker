
template <typename float>
__global__ void DropoutForward(const int n, const float* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    float* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}


template <typename float>
__global__ void DropoutBackward(const int n, const float* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    float* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}