#include "DataCollector.h"
#include "Agent.h"
#include "WorldState.h"
#include "BallState.h"
#include "PlayerState.h"
#include "ServerParam.h"
#include "Types.h"
#include <fstream>
#include <iomanip>

DataCollector::DataCollector()
    : m_last_kick_cycle(-1000)
    , m_last_kick_unum(0)
    , m_last_kick_pos_x(0.0)
    , m_dribble_start_cycle(-1000)
    , m_is_dribbling(false)
    , m_dribble_target_unum(0)
    , m_is_shooting(false)
    , m_shoot_start_cycle(-1000)
    , m_shooter_unum(0) {
    // 初始化最近射门队列
    m_recent_shoots.clear();
}

DataCollector::~DataCollector() {}

std::vector<float> DataCollector::BuildGlobalFeatures(const Agent& agent) {
    std::vector<float> features(kNumGlobalFeatures, 0.0f);

    const auto& world = agent.GetWorldState();
    const auto& self = agent.GetSelf();
    const auto& ball = world.GetBall();

    int idx = 0;

    // Ball position X/Y (2)
    features[idx++] = static_cast<float>(ball.GetPos().X());
    features[idx++] = static_cast<float>(ball.GetPos().Y());

    // Ball velocity X/Y (2)
    features[idx++] = static_cast<float>(ball.GetVel().X());
    features[idx++] = static_cast<float>(ball.GetVel().Y());

    // Self position X/Y (2)
    features[idx++] = static_cast<float>(self.GetPos().X());
    features[idx++] = static_cast<float>(self.GetPos().Y());

    // Self velocity X/Y (2)
    features[idx++] = static_cast<float>(self.GetVel().X());
    features[idx++] = static_cast<float>(self.GetVel().Y());

    // Self body direction (1)
    features[idx++] = static_cast<float>(self.GetBodyDir());

    // Self is goalkeeper (1)
    features[idx++] = static_cast<float>(self.IsGoalie());

    // Ball is kickable: self to ball distance < kickable_area (1)
    double self_to_ball_dist = self.GetPos().Dist(ball.GetPos());
    double kickable_range = ServerParam::instance().kickableArea();
    features[idx++] = static_cast<float>(self_to_ball_dist < kickable_range ? 1.0f : 0.0f);

    // Teammate 1-11 position X/Y (22)
    for (int i = 1; i <= 11; ++i) {
        const PlayerState& tm = world.GetTeammate(i);
        if (tm.IsAlive()) {
            features[idx++] = static_cast<float>(tm.GetPos().X());
            features[idx++] = static_cast<float>(tm.GetPos().Y());
        } else {
            features[idx++] = 99.0f;
            features[idx++] = 99.0f;
        }
    }

    // Teammate 1-11 velocity X/Y (22)
    for (int i = 1; i <= 11; ++i) {
        const PlayerState& tm = world.GetTeammate(i);
        if (tm.IsAlive()) {
            features[idx++] = static_cast<float>(tm.GetVel().X());
            features[idx++] = static_cast<float>(tm.GetVel().Y());
        } else {
            features[idx++] = 0.0f;
            features[idx++] = 0.0f;
        }
    }

    // Opponent 1-11 position X/Y (22)
    for (int i = 1; i <= 11; ++i) {
        const PlayerState& opp = world.GetOpponent(i);
        if (opp.IsAlive()) {
            features[idx++] = static_cast<float>(opp.GetPos().X());
            features[idx++] = static_cast<float>(opp.GetPos().Y());
        } else {
            features[idx++] = 99.0f;
            features[idx++] = 99.0f;
        }
    }

    // Opponent 1-11 velocity X/Y (22)
    for (int i = 1; i <= 11; ++i) {
        const PlayerState& opp = world.GetOpponent(i);
        if (opp.IsAlive()) {
            features[idx++] = static_cast<float>(opp.GetVel().X());
            features[idx++] = static_cast<float>(opp.GetVel().Y());
        } else {
            features[idx++] = 0.0f;
            features[idx++] = 0.0f;
        }
    }

    // Play mode (1)
    features[idx++] = static_cast<float>(world.GetPlayMode());

    // Ball ownership (1) - simplified
    features[idx++] = 0.0f;

    // Left goalkeeper position X/Y (2)
    Unum left_goalie = world.GetTeammateGoalieUnum();
    if (left_goalie > 0) {
        const PlayerState& goalie = world.GetTeammate(left_goalie);
        features[idx++] = static_cast<float>(goalie.GetPos().X());
        features[idx++] = static_cast<float>(goalie.GetPos().Y());
    } else {
        features[idx++] = -52.5f;  // default left goal center X
        features[idx++] = 0.0f;
    }

    // Left goalkeeper velocity X/Y (2)
    if (left_goalie > 0) {
        const PlayerState& goalie = world.GetTeammate(left_goalie);
        features[idx++] = static_cast<float>(goalie.GetVel().X());
        features[idx++] = static_cast<float>(goalie.GetVel().Y());
    } else {
        features[idx++] = 0.0f;
        features[idx++] = 0.0f;
    }

    // Right goalkeeper position X/Y (2)
    Unum right_goalie = world.GetOpponentGoalieUnum();
    if (right_goalie > 0) {
        const PlayerState& goalie = world.GetOpponent(right_goalie);
        features[idx++] = static_cast<float>(goalie.GetPos().X());
        features[idx++] = static_cast<float>(goalie.GetPos().Y());
    } else {
        features[idx++] = 52.5f;  // default right goal center X
        features[idx++] = 0.0f;
    }

    // Right goalkeeper velocity X/Y (2)
    if (right_goalie > 0) {
        const PlayerState& goalie = world.GetOpponent(right_goalie);
        features[idx++] = static_cast<float>(goalie.GetVel().X());
        features[idx++] = static_cast<float>(goalie.GetVel().Y());
    } else {
        features[idx++] = 0.0f;
        features[idx++] = 0.0f;
    }

    return features;
}

std::vector<float> DataCollector::BuildCandidateFeatures(int behavior_type, const Vector& target, double power) {
    std::vector<float> features(kNumCandidateFeatures, 0.0f);

    int idx = 0;

    // Behavior type (enum value) (1)
    features[idx++] = static_cast<float>(behavior_type);

    // Target position X/Y (2)
    features[idx++] = static_cast<float>(target.X());
    features[idx++] = static_cast<float>(target.Y());

    // Power (1)
    features[idx++] = static_cast<float>(power);

    return features;
}

std::vector<float> DataCollector::BuildFeatureVector(const Agent& agent, int behavior_type, const Vector& target, double power) {
    std::vector<float> features(kNumFeatures, 0.0f);

    // Build global features (109 dims)
    std::vector<float> global_features = BuildGlobalFeatures(agent);
    for (int i = 0; i < kNumGlobalFeatures; ++i) {
        features[i] = global_features[i];
    }

    // Build candidate features (4 dims)
    std::vector<float> candidate_features = BuildCandidateFeatures(behavior_type, target, power);
    for (int i = 0; i < kNumCandidateFeatures; ++i) {
        features[kNumGlobalFeatures + i] = candidate_features[i];
    }

    return features;
}

// 旧接口保留 (deprecated)
std::vector<float> DataCollector::BuildFeatureVector(const Agent& agent) {
    // 使用默认的candidate特征 (BT_None, (0,0), 0)
    return BuildFeatureVector(agent, BT_None, Vector(0, 0), 0.0);
}

// 新接口实现
void DataCollector::Collect(const Agent& agent, int behavior_type, const Vector& target, double power, float value) {
    std::vector<float> features = BuildFeatureVector(agent, behavior_type, target, power);
    Sample sample;
    sample.features = features;
    sample.value = value;
    m_samples.push_back(sample);
}

// 旧接口保留 (deprecated)
void DataCollector::Collect(const Agent& agent, float value) {
    std::vector<float> features = BuildFeatureVector(agent);
    Sample sample;
    sample.features = features;
    sample.value = value;
    m_samples.push_back(sample);
}

void DataCollector::Save(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }

    file << std::fixed << std::setprecision(6);

    for (const auto& sample : m_samples) {
        file << "{\"f\":[";
        for (size_t i = 0; i < sample.features.size(); ++i) {
            if (i > 0) file << ",";
            file << sample.features[i];
        }
        file << "],\"v\":" << sample.value << "}\n";
    }

    file.close();
}

// ========== 强化学习相关实现 ==========

void DataCollector::AddToSequence(const Agent& agent, int behavior_type, const Vector& target, double power, float nn_value) {
    // 构建完整特征向量
    std::vector<float> feature_vec = BuildFeatureVector(agent, behavior_type, target, power);

    SequenceItem item;
    item.state = feature_vec;
    item.nn_value = nn_value;
    item.behavior_type = behavior_type;
    item.action_target = target;
    item.action_power = power;
    item.rewarded = false;
    item.cycle = agent.GetWorldState().CurrentTime().T() * 100 + agent.GetWorldState().CurrentTime().S();

    m_sequence.push_back(item);

    // 保持缓冲区大小限制
    // 如果超过200，在第201次时检查最老的条目是否被reward过
    // 如果reward过就写磁盘，否则丢弃
    while (m_sequence.size() > kSequenceBufferSize) {
        if (m_sequence.front().rewarded) {
            // 写单个条目到临时缓冲区，然后清空
            static std::vector<SequenceItem> to_save;
            to_save.clear();
            to_save.push_back(m_sequence.front());
            SaveSingleItem(to_save);
        }
        m_sequence.pop_front();
    }

    static int debug_counter = 0;
    if (debug_counter++ % 100 == 0) {
        fprintf(stderr, "[AddToSequence] type=%d seq_size=%zu\n", behavior_type, m_sequence.size());
    }
}

void DataCollector::StartDribbleTracking(const Agent& agent) {
    const auto& world = agent.GetWorldState();
    const auto& self = agent.GetSelf();
    const auto& ball = world.GetBall();

    int current_cycle = world.CurrentTime().T() * 100 + world.CurrentTime().S();

    // 如果已经在追踪同一个球员的带球，不要重置
    // 这避免同一带球被多次评估
    if (m_is_dribbling && m_dribble_target_unum == self.GetUnum()) {
        return;
    }

    // 开始追踪带球
    m_is_dribbling = true;
    m_dribble_target_unum = self.GetUnum();
    m_dribble_start_cycle = current_cycle;
    m_last_kick_pos_x = ball.GetPos().X();

    fprintf(stderr, "[Dribble] Started tracking: unum=%d cycle=%d ball_x=%.2f\n",
            self.GetUnum(), current_cycle, m_last_kick_pos_x);
}

bool DataCollector::CheckDribbleSuccess(const Agent& agent) {
    const auto& world = agent.GetWorldState();
    const auto& self = agent.GetSelf();
    const auto& ball = world.GetBall();

    int current_cycle = world.CurrentTime().T() * 100 + world.CurrentTime().S();
    Unum self_unum = self.GetUnum();

    // 如果没有在带球追踪状态，直接返回
    if (!m_is_dribbling) {
        return false;
    }

    // 检查是否是在追踪这个球员的带球
    if (m_dribble_target_unum != self_unum) {
        return false;
    }

    int cycles_since_dribble = current_cycle - m_dribble_start_cycle;

    // 调试输出（每100帧输出一次）
    static int debug_counter = 0;
    if (debug_counter++ % 100 == 0) {
        double self_to_ball_dist = self.GetPos().Dist(ball.GetPos());
        double kickable_range = ServerParam::instance().kickableArea();
        fprintf(stderr, "[DribbleDebug] cycle=%d unum=%d start_cycle=%d cycles_since=%d dist=%.2f kickable=%.2f\n",
                current_cycle, self_unum, m_dribble_start_cycle, cycles_since_dribble,
                self_to_ball_dist, kickable_range);
    }

    // 带球追踪15个周期后评估结果（只评估一次）
    if (cycles_since_dribble >= 15) {
        float dribble_reward;
        double dribble_distance = ball.GetPos().X() - m_last_kick_pos_x;

        // 使用与 BuildGlobalFeatures 相同的判断标准
        double self_to_ball_dist = self.GetPos().Dist(ball.GetPos());
        double kickable_range = ServerParam::instance().kickableArea();
        bool near_ball = self_to_ball_dist < kickable_range;

        if (near_ball) {
            // 还控着球，视为成功
            dribble_reward = 1.0f;
            fprintf(stderr, "[Dribble] SUCCESS! reward=%.1f cycles=%d\n",
                    dribble_reward, cycles_since_dribble);
        } else {
            // 失去球权，视为失败
            dribble_reward = -1.0f;
            fprintf(stderr, "[Dribble] FAILURE! penalty=%.1f cycles=%d\n",
                    dribble_reward, cycles_since_dribble);
        }

        // 给予奖励到序列
        RewardSequence(dribble_reward);

        // 重置带球追踪状态
        m_is_dribbling = false;
        m_dribble_start_cycle = -1000;
        m_dribble_target_unum = 0;

        return near_ball;
    }

    return false;
}

// ========== 射门强化学习实现 ==========

void DataCollector::StartShootTracking(const Agent& agent, const Vector& target) {
    const auto& world = agent.GetWorldState();
    const auto& self = agent.GetSelf();
    const auto& ball = world.GetBall();

    int current_cycle = world.CurrentTime().T() * 100 + world.CurrentTime().S();

    // 如果已经在追踪射门，不要重置（避免同一射门被多次评估）
    if (m_is_shooting) {
        return;
    }

    // 开始追踪射门
    m_is_shooting = true;
    m_shoot_start_cycle = current_cycle;
    m_shoot_target = target;
    m_shoot_ball_start_pos = ball.GetPos();
    m_shooter_unum = self.GetUnum();

    fprintf(stderr, "[Shoot] Started tracking: unum=%d cycle=%d target=(%.2f,%.2f) ball=(%.2f,%.2f)\n",
            self.GetUnum(), current_cycle, target.X(), target.Y(), ball.GetPos().X(), ball.GetPos().Y());
}

bool DataCollector::CheckShootResult(const Agent& agent) {
    const auto& world = agent.GetWorldState();
    const auto& ball = world.GetBall();

    int current_cycle = world.CurrentTime().T() * 100 + world.CurrentTime().S();

    // 如果没有在追踪射门，直接返回
    if (!m_is_shooting) {
        return false;
    }

    // 射门追踪25个周期后评估结果（只评估一次）
    // 25周期足够让球飞到球门并产生结果
    int cycles_since_shoot = current_cycle - m_shoot_start_cycle;
    if (cycles_since_shoot < 25) {
        return false;
    }

    float shoot_reward = 0.0f;
    bool evaluated = false;
    bool is_goal = false;

    // 检查进球：play mode变成我方进球
    PlayMode pm = world.GetPlayMode();
    if (pm == PM_Goal_Ours) {
        // 进球了！正面奖励
        shoot_reward = 1.0f;
        is_goal = true;
        evaluated = true;
        fprintf(stderr, "[Shoot] GOAL! reward=%.3f cycles=%d\n", shoot_reward, cycles_since_shoot);
    } else {
        // 25周期后比赛模式没变化，说明射门不成功 - 负奖励
        shoot_reward = -1.0f;
        is_goal = false;
        evaluated = true;
        fprintf(stderr, "[Shoot] MISS (no goal). reward=%.3f cycles=%d ball=(%.2f,%.2f)\n",
                shoot_reward, cycles_since_shoot, ball.GetPos().X(), ball.GetPos().Y());
    }

    // 追踪最近射门成功/失败次数
    m_recent_shoots.push_back(is_goal);
    if (m_recent_shoots.size() > 10) {
        m_recent_shoots.pop_front();
    }

    if (evaluated) {
        // 给予奖励到序列（只奖励射门相关的序列条目）
        // 使用较长的回溯窗口，因为射门决策可能较早
        RewardSequence(shoot_reward, 30);

        // 重置射门追踪状态
        m_is_shooting = false;
        m_shoot_start_cycle = -1000;
        m_shooter_unum = 0;

        return true;
    }

    // 30周期后还没确定结果，继续等待（罕见情况）
    return false;
}

void DataCollector::RewardSequence(float reward, int max_back_cycles) {
    // 从序列末尾向前搜索，最多回溯max_back_cycles个周期
    // 标记为rewarded，并存储reward值
    // TD target 计算: V_target = r + γ * V(s')

    int current_cycle = 0;
    if (!m_sequence.empty()) {
        current_cycle = m_sequence.back().cycle;
    }

    int marked_count = 0;
    for (auto it = m_sequence.rbegin(); it != m_sequence.rend(); ++it) {
        int cycle_diff = current_cycle - it->cycle;
        if (cycle_diff > max_back_cycles * 100) {
            break;
        }

        if (!it->rewarded) {
            it->rewarded = true;
            it->received_reward = reward;  // 存储奖励值
            marked_count++;
        }
    }

    fprintf(stderr, "[RewardSequence] reward=%.3f marked=%d seq_size=%zu\n",
            reward, marked_count, m_sequence.size());
}

void DataCollector::MarkLastActionExecuted() {
    // 标记最后一个动作被执行了（用于过滤未被执行的动作）
    // 目前在RewardSequence中我们不过滤未执行的动作，
    // 因为NN的价值估计本身就是对好动作的评估
}

int DataCollector::GetRewardedCount() const {
    int count = 0;
    for (const auto& item : m_sequence) {
        if (item.rewarded) {
            count++;
        }
    }
    return count;
}

void DataCollector::ResetSequence() {
    m_sequence.clear();
    m_last_kick_cycle = -1000;
    m_last_kick_unum = 0;
    m_last_kick_pos_x = 0.0;
    m_dribble_start_cycle = -1000;
    m_is_dribbling = false;
    m_dribble_target_unum = 0;
    // 重置射门追踪状态
    m_is_shooting = false;
    m_shoot_start_cycle = -1000;
    m_shooter_unum = 0;
    // 清空最近射门记录
    m_recent_shoots.clear();
}

void DataCollector::SaveSequence(const std::string& filename) {
    int rewarded_count = GetRewardedCount();
    fprintf(stderr, "[SaveSequence] filename=%s rewarded_count=%d seq_size=%zu\n",
            filename.c_str(), rewarded_count, m_sequence.size());

    if (rewarded_count == 0) {
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }

    file << std::fixed << std::setprecision(6);

    // TD target 计算: V(s_t) = r_t + γ * V(s_{t+1})
    // 带球成功时，只有最后一个状态收到 r = 0.2，其他状态 r = 0
    // γ = 0.9（折扣因子）

    const float gamma = 0.9f;
    // 从后向前计算 TD target
    // 使用存储在item中的received_reward值
    float next_value = 0.0f;  // 下一个状态的 V 值（初始为0）
    bool first = true;
    int saved_count = 0;

    for (auto it = m_sequence.rbegin(); it != m_sequence.rend(); ++it) {
        if (!it->rewarded) {
            continue;
        }

        // TD target = received_reward + γ * V(s')
        float td_target;
        if (first) {
            // 最后一个（最近的）rewarded 状态: r = received_reward
            td_target = it->received_reward + gamma * next_value;
            first = false;
        } else {
            // 之前的 rewarded 状态: r = 0
            td_target = gamma * next_value;
        }

        // 保存这个条目的数据
        file << "{\"f\":[";
        for (size_t j = 0; j < it->state.size(); ++j) {
            if (j > 0) file << ",";
            file << it->state[j];
        }
        file << "],\"v\":" << td_target << "}\n";
        saved_count++;

        // 更新下一个状态的 TD target
        next_value = td_target;
    }

    file.close();
    fprintf(stderr, "[SaveSequence] saved %d items\n", saved_count);
}

// 保存单个条目到增量文件（用于缓冲区溢出时）
void DataCollector::SaveSingleItem(const std::vector<SequenceItem>& items) {
    if (items.empty()) {
        return;
    }

    // 使用固定的增量文件名
    static std::string incremental_filename = "Logfiles/rl_data_incremental.json";

    std::ofstream file(incremental_filename, std::ios::app);
    if (!file.is_open()) {
        fprintf(stderr, "[SaveSingleItem] Failed to open %s\n", incremental_filename.c_str());
        return;
    }

    file << std::fixed << std::setprecision(6);

    // 使用存储在item中的reward值作为TD target
    for (const auto& item : items) {
        float td_target = item.received_reward;  // 使用存储的奖励值

        file << "{\"f\":[";
        for (size_t j = 0; j < item.state.size(); ++j) {
            if (j > 0) file << ",";
            file << item.state[j];
        }
        file << "],\"v\":" << td_target << "}\n";
    }

    file.close();
    fprintf(stderr, "[SaveSingleItem] saved %zu items to incremental file\n", items.size());
}
