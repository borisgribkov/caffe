#include <algorithm>
#include <vector>

#include "caffe/layers/combined_angular_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void CombinedAngularMarginLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const CombinedAngularMarginParameter& param = this->layer_param_.combined_angular_margin_param();
    angle_ = param.angle();
    margin_ = param.margin();
  }

  template <typename Dtype>
  void CombinedAngularMarginLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    if(top[0] != bottom[0]) top[0]->ReshapeLike(*bottom[0]);
  }

template <typename Dtype>
void CombinedAngularMarginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  Dtype cos_m = std::cos(angle_);
  Dtype sin_m = std::sin(angle_);

  caffe_copy(count, bottom_data, top_data);
  
  for (int i = 0; i < num; ++i) {
    int gt = static_cast<int>(label_data[i]);
    Dtype bottom_val = bottom_data[i * dim + gt];
    Dtype sin_t = std::sqrt(1 - bottom_val * bottom_val);
    top_data[i * dim + gt] = bottom_val * cos_m - sin_t * sin_m;
    top_data[i * dim + gt] -= margin_;
  }
}

template <typename Dtype>
void CombinedAngularMarginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  Dtype cos_m = std::cos(angle_);
  Dtype sin_m = std::sin(angle_);
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_copy(count, top_diff, bottom_diff);
  for (int i = 0; i < num; ++i) {
    int gt = static_cast<int>(label_data[i]);
    Dtype bottom_val = bottom_data[i * dim + gt];
    Dtype sin_t = std::sqrt(1 - bottom_val * bottom_val);
    bottom_diff[i * dim + gt] *= cos_m + sin_m * bottom_val / sin_t;
  }
}


#ifdef CPU_ONLY
STUB_GPU(CombinedAngularMarginLayer);
#endif

INSTANTIATE_CLASS(CombinedAngularMarginLayer);
REGISTER_LAYER_CLASS(CombinedAngularMargin);

}  // namespace caffe