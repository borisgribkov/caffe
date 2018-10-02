#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/sun_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SunLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  beta_ = this->layer_param_.sun_loss_param().beta();
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.sun_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
}

template <typename Dtype>
void SunLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = bottom[0]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Dtype>::Reshape(bottom, top);
  s_normalized_.Reshape({ K_ });
  cos_distance_.Reshape({ M_ });
}

template <typename Dtype>
void SunLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype loss = Dtype(0);
  caffe_set(K_, Dtype(0), s_normalized_.mutable_cpu_data());
  for (int i = 0; i < M_; i++) {
    caffe_add(K_, s_normalized_.cpu_data(), bottom_data + i * K_, s_normalized_.mutable_cpu_data());
  }
  caffe_scal<Dtype>(K_, Dtype(1) / M_, s_normalized_.mutable_cpu_data());
  Dtype s_norm = caffe_cpu_dot(K_, s_normalized_.cpu_data(), s_normalized_.cpu_data());
  caffe_scal<Dtype>(K_, Dtype(1) / sqrt(s_norm), s_normalized_.mutable_cpu_data());
  for (int i = 0; i < M_; i++) {
    cos_distance_.mutable_cpu_data()[i] = caffe_cpu_dot(K_, bottom_data + i * K_, s_normalized_.cpu_data());
    loss += std::max(Dtype(0), cos_distance_.cpu_data()[i] - beta_);
  }
  loss /= M_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SunLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    for (int i = 0; i < M_; i++) {
      if (cos_distance_.cpu_data()[i] < beta_) {
        caffe_set(K_, Dtype(0), bottom[0]->mutable_cpu_diff() + i * K_);
      } else {
        caffe_copy(K_, s_normalized_.cpu_data(), bottom[0]->mutable_cpu_diff() + i * K_);
        caffe_axpy(K_, -cos_distance_.cpu_data()[i], bottom[0]->cpu_data() + i * K_, bottom[0]->mutable_cpu_diff() + i * K_);
      }
    }
    caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / M_, bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(SunLossLayer);
#endif

INSTANTIATE_CLASS(SunLossLayer);
REGISTER_LAYER_CLASS(SunLoss);

}  // namespace caffe