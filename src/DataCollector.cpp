#include "DataCollector.h"
#include "Agent.h"
#include "WorldState.h"
#include "BallState.h"
#include "PlayerState.h"
#include "ServerParam.h"
#include <fstream>
#include <iomanip>

DataCollector::DataCollector() {}
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
