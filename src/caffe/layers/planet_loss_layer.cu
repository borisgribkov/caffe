#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/planet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_forward(const int nthreads, const int K,
                               Dtype* p,
                               Dtype* p_norm, 
                               const Dtype* bottom, 
                               const Dtype* label,
                               Dtype* cos_distance, 
                               Dtype* p_update_cnt, 
                               Dtype epsilon) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    p_update_cnt[label_value] += Dtype(1);
    Dtype dot = 0;
    for (int d = 0; d < K; ++d) {
      dot += p[label_value * K + d] * p[label_value * K + d];
    }
    for (int d = 0; d < K; ++d) {
      p_norm[index * K + d] = p[label_value * K + d] / sqrt(dot + epsilon);
      //p[label_value * K + d] = p_norm[index * K + d];
      cos_distance[index] += p_norm[index * K + d] * bottom[index * K + d];
    }
  }
}

template <typename Dtype>
__global__ void kernel_update_p(int nthreads, const int K, 
                                const Dtype* bottom, 
                                const Dtype* label,
                                const Dtype* p_update_cnt,
                              	Dtype* p,
                                Dtype alpha) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    p[label_value * K + k] += bottom[index] * alpha / p_update_cnt[label_value];
  }
}

template <typename Dtype>
__global__ void kernel_backward(int nthreads, const int K, 
                                const Dtype* bottom, 
                                const Dtype* p_norm,
                                const Dtype* cos_distance,
                                Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    bottom_diff[index] = p_norm[index] - cos_distance[m] * bottom[index];
  }
}

template <typename Dtype>
void PlanetLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int nthreads = M_ * K_;
  caffe_gpu_set(N_, Dtype(0), p_update_cnt_.mutable_gpu_data());
  caffe_gpu_set(M_, Dtype(0), cos_distance_.mutable_gpu_data());
  kernel_forward<Dtype><<<CAFFE_GET_BLOCKS(M_), 
      CAFFE_CUDA_NUM_THREADS>>>(M_, K_, 
        this->blobs_[0]->mutable_gpu_data(), 
        p_normalized_.mutable_gpu_data(),
        bottom[0]->gpu_data(),
        bottom[1]->gpu_data(),
        cos_distance_.mutable_gpu_data(),
        p_update_cnt_.mutable_gpu_data(),
        1e-12);
  Dtype dot;
  caffe_gpu_dot(M_ * K_, bottom[0]->gpu_data(), p_normalized_.gpu_data(), &dot);
  Dtype loss = Dtype(1) - dot / M_;
  top[0]->mutable_cpu_data()[0] = loss;
  kernel_update_p<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, 
        bottom[2]->gpu_data(),
        bottom[1]->gpu_data(),
        p_update_cnt_.gpu_data(),
        this->blobs_[0]->mutable_gpu_data(), 
        alpha_);
}

template <typename Dtype>
void PlanetLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int nthreads = M_ * K_;
  if (propagate_down[0]) {
    kernel_backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, 
        bottom[0]->gpu_data(),
        p_normalized_.gpu_data(),
        cos_distance_.gpu_data(),
        bottom[0]->mutable_gpu_diff());
    caffe_gpu_scale(M_ * K_, -(top[0]->cpu_diff()[0] / M_), 
                             bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PlanetLossLayer);

}  // namespace caffe