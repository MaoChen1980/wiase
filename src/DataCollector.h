#ifndef DATACOLLECTOR_H_
#define DATACOLLECTOR_H_

#include <vector>
#include <string>
#include <deque>
#include "Geometry.h"

class Agent;

class DataCollector {
public:
    // 特征数量：全局(109) + candidate(4) = 113
    static const int kNumFeatures = 113;
    static const int kNumGlobalFeatures = 109;
    static const int kNumCandidateFeatures = 4;

    // RL序列缓冲区大小
    static const int kSequenceBufferSize = 200;

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

    // ========== 强化学习相关 ==========

    // 序列中的单个条目
    struct SequenceItem {
        std::vector<float> state;      // 113维状态
        float nn_value;                // NN的评估值
        int behavior_type;             // 行为类型
        Vector action_target;           // 行为目标
        double action_power;            // 行为力量
        bool rewarded;                  // 是否已被reward
        float received_reward;          // 收到的奖励值（仅在rewarded时有效）
        int cycle;                      // 周期号
    };

    // 添加到序列缓冲区
    void AddToSequence(const Agent& agent, int behavior_type, const Vector& target, double power, float nn_value);

    // 开始追踪带球（当带球被选中时调用）
    void StartDribbleTracking(const Agent& agent);

    // 检查并处理带球成功 - 返回是否发生了奖励事件
    bool CheckDribbleSuccess(const Agent& agent);

    // ========== 射门强化学习相关 ==========

    // 开始追踪射门（当射门被选中时调用）
    void StartShootTracking(const Agent& agent, const Vector& target);

    // 检查并处理射门结果 - 返回是否发生了奖励事件
    bool CheckShootResult(const Agent& agent);

    // 给予奖励到序列中的相关条目
    void RewardSequence(float reward, int max_back_cycles = 20);

    // 标记当前序列中的行为被执行了
    void MarkLastActionExecuted();

    // 获取序列中需要保存的条目数
    int GetRewardedCount() const;

    // 保存到文件 - 只保存被reward过的
    void Save(const std::string& filename);

    // 保存RL序列数据到文件 - 只保存被reward过的
    void SaveSequence(const std::string& filename);

    // 保存单个条目（用于缓冲区溢出时保存已reward的条目）
    void SaveSingleItem(const std::vector<SequenceItem>& items);

    // 获取已收集的样本数
    int NumSamples() const { return m_samples.size(); }

    // 重置序列缓冲区
    void ResetSequence();

    // 获取缓冲区大小
    int GetSequenceSize() const { return m_sequence.size(); }

private:
    struct Sample {
        std::vector<float> features;
        float value;
    };

    std::vector<Sample> m_samples;

    // RL序列缓冲区
    std::deque<SequenceItem> m_sequence;

    // 用于检测带球的上一次踢球信息
    int m_last_kick_cycle;         // 上次踢球时的周期
    int m_last_kick_unum;          // 上次踢球的是谁
    float m_last_kick_pos_x;       // 上次踢球后的球位置X
    int m_dribble_start_cycle;     // 带球开始时的周期

    // 用于追踪带球状态
    bool m_is_dribbling;            // 是否正在带球
    int m_dribble_target_unum;     // 带球球员编号

    // 用于追踪射门状态
    bool m_is_shooting;            // 是否正在追踪射门
    int m_shoot_start_cycle;       // 射门开始时的周期
    Vector m_shoot_target;         // 射门目标
    Vector m_shoot_ball_start_pos; // 射门时球的位置
    Unum m_shooter_unum;          // 射门球员编号

    // 最近射门结果（用于统计）
    std::deque<bool> m_recent_shoots;  // 最近10次射门是否进球
};

#endif