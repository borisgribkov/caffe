#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/planet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PlanetLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  alpha_ = this->layer_param_.planet_loss_param().alpha();
  const int num_output = this->layer_param_.planet_loss_param().num_output();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.planet_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> p_shape(2);
    p_shape[0] = N_;
    p_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(p_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > p_filler(GetFiller<Dtype>(
        this->layer_param_.planet_loss_param().p_filler()));
    p_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), false);
}

template <typename Dtype>
void PlanetLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  M_ = bottom[0]->num();
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  LossLayer<Dtype>::Reshape(bottom, top);
  p_update_cnt_.Reshape({ N_ });
  p_normalized_.Reshape({ M_, K_ });
  cos_distance_.Reshape({ M_ });
}

template <typename Dtype>
void PlanetLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* p = this->blobs_[0]->cpu_data();
  Dtype* p_mutable = this->blobs_[0]->mutable_cpu_data();
  Dtype cos_sum = Dtype(0);
  caffe_set(N_, Dtype(0), p_update_cnt_.mutable_cpu_data());
  for (int i = 0; i < M_; i++) {
    const int label_value = static_cast<int>(label[i]);
    caffe_copy(K_, p + label_value * K_, p_normalized_.mutable_cpu_data() + i * K_);
    Dtype p_norm = caffe_cpu_dot(K_, p_normalized_.cpu_data() + i * K_, p_normalized_.cpu_data() + i * K_) + static_cast<Dtype>(1e-12);
    caffe_scal<Dtype>(K_, Dtype(1) / sqrt(p_norm), p_normalized_.mutable_cpu_data() + i * K_);
    cos_distance_.mutable_cpu_data()[i] = caffe_cpu_dot(K_, bottom_data + i * K_, p_normalized_.cpu_data() + i * K_);
    cos_sum += cos_distance_.mutable_cpu_data()[i];
    p_update_cnt_.mutable_cpu_data()[label_value] += Dtype(1);
    caffe_copy(K_, p_normalized_.cpu_data() + i * K_, p_mutable + label_value * K_);
  }
  cos_sum /= M_;
  top[0]->mutable_cpu_data()[0] = Dtype(1) - cos_sum;
  for (int i = 0; i < M_; ++i) {
    const int label_value = static_cast<int>(label[i]);
    caffe_axpy(K_, alpha_/p_update_cnt_.cpu_data()[label_value], bottom_data + i * K_, p_mutable + label_value * K_);
  }
}

template <typename Dtype>
void PlanetLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    for (int i = 0; i < M_; i++) {
      caffe_copy(K_, p_normalized_.cpu_data() + i * K_, bottom[0]->mutable_cpu_diff() + i * K_);
      caffe_axpy(K_, -cos_distance_.cpu_data()[i], bottom[0]->cpu_data() + i * K_, bottom[0]->mutable_cpu_diff() + i * K_);
    }
    caffe_scal(M_ * K_, -(top[0]->cpu_diff()[0] / M_), bottom[0]->mutable_cpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(PlanetLossLayer);
#endif

INSTANTIATE_CLASS(PlanetLossLayer);
REGISTER_LAYER_CLASS(PlanetLoss);

}  // namespace caffe