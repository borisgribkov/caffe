#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kl_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KLLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void KLLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Dimensions don't match";
}

template <typename Dtype>
void KLLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob1 = prob_.cpu_data();
  const Dtype* prob2 = bottom[1]->cpu_data();
  Dtype * temp = bottom[0]->mutable_cpu_diff();
  Dtype loss = 0;
  for (int i = 0; i < prob_.count(); ++i) {
    temp[i] = std::max(prob2[i], Dtype(FLT_MIN));
  }
  caffe_log(prob_.count(), temp, temp);
  for (int i = 0; i < prob_.count(); ++i) {
    temp[i] -= log(std::max(prob1[i], Dtype(FLT_MIN)));
  }
  loss = caffe_cpu_dot(prob_.count(), temp, prob2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void KLLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob1 = prob_.cpu_data();
    const Dtype* prob2 = bottom[1]->cpu_data();
    caffe_sub(prob_.count(), prob1, prob2, bottom_diff);
    Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(KLLossLayer);
#endif

INSTANTIATE_CLASS(KLLossLayer);
REGISTER_LAYER_CLASS(KLLoss);

}  // namespace caffe
