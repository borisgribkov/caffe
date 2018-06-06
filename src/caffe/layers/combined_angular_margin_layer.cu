#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/combined_angular_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void CombinedAngularMarginForward(const int n, const int dim, const Dtype* label,
                                                 const Dtype* bottom_data, Dtype* top_data, Dtype angle, Dtype margin, Dtype scale) {
    Dtype cos_m = cosf(angle);
    Dtype sin_m = sinf(angle);
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      Dtype cos_t = bottom_data[index * dim + gt] / scale;
      Dtype sin_t = sqrtf(Dtype(1) - cos_t * cos_t);
      top_data[index * dim + gt] = (cos_t * cos_m - sin_t * sin_m - margin) * scale;
    }
  }

  template <typename Dtype>
  __global__ void CombinedAngularMarginBackward(const int n, const int dim, const Dtype* label,
                                                 const Dtype* bottom_data, Dtype* bottom_diff, Dtype angle, Dtype scale) {
    Dtype cos_m = cosf(angle);
    Dtype sin_m = sinf(angle);
    CUDA_KERNEL_LOOP(index, n) {
      int gt = static_cast<int>(label[index]);
      Dtype cos_t = bottom_data[index * dim + gt] / scale;
      Dtype sin_t = sqrtf(Dtype(1) - cos_t * cos_t);
      bottom_diff[index * dim + gt] *= cos_m + sin_m * cos_t / sin_t;
    }
  }

  template <typename Dtype>
  void CombinedAngularMarginLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    caffe_copy(count, bottom_data, top_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    CombinedAngularMarginForward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, label_data, bottom_data, top_data, angle_, margin_, scale_);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  void CombinedAngularMarginLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down,
                                                     const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) { return; }
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    // NOLINT_NEXT_LINE(whitespace/operators)
    CombinedAngularMarginBackward<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> > (
      num, dim, label_data, bottom_data, bottom_diff, angle_, scale_);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(CombinedAngularMarginLayer);
}  // namespace caffe