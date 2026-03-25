#ifndef NNVALUEINFERENCE_H_
#define NNVALUEINFERENCE_H_

#include <vector>
#include <string>

class Agent;

class NNValueInference {
public:
    NNValueInference();
    ~NNValueInference();
    
    // 加载模型
    bool Load(const std::string& filename);
    
    // 推理：输入特征，输出价值
    float Forward(const std::vector<float>& features);

    // 检查模型是否已加载
    bool Loaded() const { return loaded_; }
    
private:
    // 模型参数
    static const int kInputDim = 113;
    static const int kHidden1 = 64;
    static const int kHidden2 = 32;
    static const int kHidden3 = 16;

    std::vector<float> mean_;
    std::vector<float> std_;

    // 权重
    std::vector<float> w1_;  // (64, 113)
    std::vector<float> b1_;  // (64)
    std::vector<float> w2_;  // (32, 64)
    std::vector<float> b2_;  // (32)
    std::vector<float> w3_;  // (16, 32)
    std::vector<float> b3_;  // (16)
    std::vector<float> w4_;  // (1, 16)
    std::vector<float> b4_;  // (1)
    
    bool loaded_;
    
    float Relu(float x) { return x > 0 ? x : 0; }
};

#endif
