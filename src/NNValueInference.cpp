#include "NNValueInference.h"
#include <fstream>
#include <cstring>

NNValueInference::NNValueInference() : loaded_(false) {}

NNValueInference::~NNValueInference() {}

bool NNValueInference::Load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "[NN] Failed to open %s\n", filename.c_str());
        return false;
    }

    // 读取 magic
    int magic;
    file.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 1) {
        fprintf(stderr, "[NN] Invalid magic: %d\n", magic);
        return false;
    }

    // 读取 mean, std (113 dims)
    mean_.resize(kInputDim);
    std_.resize(kInputDim);
    file.read(reinterpret_cast<char*>(mean_.data()), kInputDim * sizeof(float));
    file.read(reinterpret_cast<char*>(std_.data()), kInputDim * sizeof(float));

    // 读取权重: 113 -> 64 -> 32 -> 16 -> 1
    // w1: (64, 113), flatten row-major
    w1_.resize(kHidden1 * kInputDim);
    b1_.resize(kHidden1);
    // w2: (32, 64)
    w2_.resize(kHidden2 * kHidden1);
    b2_.resize(kHidden2);
    // w3: (16, 32)
    w3_.resize(kHidden3 * kHidden2);
    b3_.resize(kHidden3);
    // w4: (1, 16)
    w4_.resize(kHidden3);  // 1 * 16
    b4_.resize(1);

    file.read(reinterpret_cast<char*>(w1_.data()), kHidden1 * kInputDim * sizeof(float));
    file.read(reinterpret_cast<char*>(b1_.data()), kHidden1 * sizeof(float));
    file.read(reinterpret_cast<char*>(w2_.data()), kHidden2 * kHidden1 * sizeof(float));
    file.read(reinterpret_cast<char*>(b2_.data()), kHidden2 * sizeof(float));
    file.read(reinterpret_cast<char*>(w3_.data()), kHidden3 * kHidden2 * sizeof(float));
    file.read(reinterpret_cast<char*>(b3_.data()), kHidden3 * sizeof(float));
    file.read(reinterpret_cast<char*>(w4_.data()), kHidden3 * sizeof(float));
    file.read(reinterpret_cast<char*>(b4_.data()), sizeof(float));

    file.close();
    loaded_ = true;
    fprintf(stderr, "[NN] Model loaded: 113->64->32->16->1\n", filename.c_str());
    return true;
}

float NNValueInference::Forward(const std::vector<float>& features) {
    if (!loaded_ || features.size() != static_cast<size_t>(kInputDim)) {
        return 0.0f;
    }

    // 归一化
    std::vector<float> x(kInputDim);
    for (int i = 0; i < kInputDim; i++) {
        x[i] = (features[i] - mean_[i]) / (std_[i] + 1e-8);
    }

    // 第一层: 113 -> 64
    // Python w1_flat[j*113 + i] = weight[j, i], weight shape: (64, 113)
    std::vector<float> h1(kHidden1, 0);
    for (int j = 0; j < kHidden1; j++) {
        float sum = b1_[j];
        for (int i = 0; i < kInputDim; i++) {
            sum += x[i] * w1_[j * kInputDim + i];
        }
        h1[j] = Relu(sum);
    }

    // 第二层: 64 -> 32
    // w2_flat[j*64 + i] = weight[j, i], weight shape: (32, 64)
    std::vector<float> h2(kHidden2, 0);
    for (int j = 0; j < kHidden2; j++) {
        float sum = b2_[j];
        for (int i = 0; i < kHidden1; i++) {
            sum += h1[i] * w2_[j * kHidden1 + i];
        }
        h2[j] = Relu(sum);
    }

    // 第三层: 32 -> 16
    // w3_flat[j*32 + i] = weight[j, i], weight shape: (16, 32)
    std::vector<float> h3(kHidden3, 0);
    for (int j = 0; j < kHidden3; j++) {
        float sum = b3_[j];
        for (int i = 0; i < kHidden2; i++) {
            sum += h2[i] * w3_[j * kHidden2 + i];
        }
        h3[j] = Relu(sum);
    }

    // 第四层: 16 -> 1
    // w4_flat[j*16 + i] = weight[j, i], weight shape: (1, 16)
    float sum = b4_[0];
    for (int i = 0; i < kHidden3; i++) {
        sum += h3[i] * w4_[i];
    }

    return sum;
}
