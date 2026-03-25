#ifndef DATACOLLECTOR_H_
#define DATACOLLECTOR_H_

#include <vector>
#include <string>

class Agent;
class Vector;

class DataCollector {
public:
    // 特征数量：全局(109) + candidate(4) = 113
    static const int kNumFeatures = 113;
    static const int kNumGlobalFeatures = 109;
    static const int kNumCandidateFeatures = 4;

    DataCollector();
    ~DataCollector();

    // 构建全局特征向量 (109维)
    std::vector<float> BuildGlobalFeatures(const Agent& agent);

    // 构建候选行为特征向量 (4维)
    std::vector<float> BuildCandidateFeatures(int behavior_type, const Vector& target, double power);

    // 构建完整特征向量 (全局+候选 = 113维)
    std::vector<float> BuildFeatureVector(const Agent& agent, int behavior_type, const Vector& target, double power);

    // 旧接口保留 (deprecated)
    std::vector<float> BuildFeatureVector(const Agent& agent);

    // 收集数据 - 带target和power
    void Collect(const Agent& agent, int behavior_type, const Vector& target, double power, float value);

    // 旧接口保留 (deprecated)
    void Collect(const Agent& agent, float value);
    
    // 保存到文件
    void Save(const std::string& filename);
    
    // 获取已收集的样本数
    int NumSamples() const { return m_samples.size(); }
    
private:
    struct Sample {
        std::vector<float> features;
        float value;
    };
    
    std::vector<Sample> m_samples;
};

#endif
