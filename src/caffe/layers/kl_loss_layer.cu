#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kl_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kl_forward_gpu(const int N,
          const Dtype* prob1, const Dtype* prob2, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, N) {
    loss[index] = log(max(prob2[index],Dtype(FLT_MIN)));
    loss[index] -= log(max(prob1[index],Dtype(FLT_MIN)));
  }
}

template <typename Dtype>
void KLLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob1 = prob_.gpu_data();
  const Dtype* prob2 = bottom[1]->gpu_data();
  Dtype* temp = bottom[0]->mutable_gpu_diff();

  Dtype loss = 0;
  kl_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(prob_.count()),CAFFE_CUDA_NUM_THREADS>>>(prob_.count(), prob1, prob2, temp);
  caffe_gpu_dot(prob_.count(), temp, prob2, &loss);

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void KLLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to target distribution yet.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob1 = prob_.gpu_data();
    const Dtype* prob2 = bottom[1]->gpu_data();
    
    caffe_gpu_sub(prob_.count(), prob1, prob2, bottom_diff);
    Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KLLossLayer);

}  // namespace caffe