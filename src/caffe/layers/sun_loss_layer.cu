#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/sun_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_compute_center(const int nthreads, const int K, const int M,
                               const Dtype* bottom, 
                               Dtype* center) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    center[index] = Dtype(0);
    for (int m = 0; m < M; ++m) {
      center[index] += bottom[m * K + index];
    }
    center[index] /= M;
  }
}

template <typename Dtype>
__global__ void kernel_backward(int nthreads, const int K, 
                                const Dtype* bottom, 
                                const Dtype* s_normalized,
                                const Dtype* cos_distance,
                                Dtype* bottom_diff,
                                Dtype beta) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    if (cos_distance[m] < beta) {
      bottom_diff[index] = Dtype(0);
    } else {
      bottom_diff[index] = s_normalized[k] - cos_distance[m] * bottom[index];
    }
  }
}

template <typename Dtype>
void SunLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_gpu_set(K_, Dtype(0), s_normalized_.mutable_gpu_data());
  caffe_gpu_set(M_, Dtype(0), cos_distance_.mutable_gpu_data());
  for (int i = 0; i < M_; ++i) {
    caffe_gpu_add(K_, s_normalized_.gpu_data(), bottom[0]->gpu_data() + i * K_, s_normalized_.mutable_gpu_data());
  }
  /*kernel_compute_center<Dtype><<<CAFFE_GET_BLOCKS(K_), 
      CAFFE_CUDA_NUM_THREADS>>>(K_, K_, M_,
        bottom[0]->gpu_data(),
        s_normalized_.mutable_gpu_data());
  */
  caffe_gpu_scale(K_, Dtype(1) / M_, s_normalized_.gpu_data(), s_normalized_.mutable_gpu_data());
  Dtype s_norm;
  caffe_gpu_dot(K_, s_normalized_.gpu_data(), s_normalized_.gpu_data(), &s_norm);
  s_norm += Dtype(1e-12);
  caffe_gpu_scale(K_, Dtype(1) / sqrt(s_norm), s_normalized_.gpu_data(), s_normalized_.mutable_gpu_data());
  Dtype loss = Dtype(0);
  for (int i = 0; i < M_; ++i) {
    Dtype cos;
    caffe_gpu_dot(K_, bottom[0]->gpu_data() + i * K_, s_normalized_.gpu_data(), &cos);
    caffe_gpu_set(1, cos, cos_distance_.mutable_gpu_data() + i);
    loss += max(Dtype(0), cos - beta_);
  }
  loss /= M_;

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SunLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int nthreads = M_ * K_;
  if (propagate_down[0]) {
    kernel_backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, 
        bottom[0]->gpu_data(),
        s_normalized_.gpu_data(),
        cos_distance_.gpu_data(),
        bottom[0]->mutable_gpu_diff(),
        beta_);
    caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
                             bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SunLossLayer);

}  // namespace caffe