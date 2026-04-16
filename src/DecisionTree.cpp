/************************************************************************************
 * WrightEagle (Soccer Simulation League 2D) * BASE SOURCE CODE RELEASE 2016 *
 * Copyright (c) 1998-2016 WrightEagle 2D Soccer Simulation Team, * Multi-Agent
 *Systems Lab.,                                * School of Computer Science and
 *Technology,               * University of Science and Technology of China *
 * All rights reserved. *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without *
 * modification, are permitted provided that the following conditions are met: *
 *     * Redistributions of source code must retain the above copyright *
 *       notice, this list of conditions and the following disclaimer. *
 *     * Redistributions in binary form must reproduce the above copyright *
 *       notice, this list of conditions and the following disclaimer in the *
 *       documentation and/or other materials provided with the distribution. *
 *     * Neither the name of the WrightEagle 2D Soccer Simulation Team nor the *
 *       names of its contributors may be used to endorse or promote products *
 *       derived from this software without specific prior written permission. *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED    * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *PURPOSE ARE           * DISCLAIMED. IN NO EVENT SHALL WrightEagle 2D Soccer
 *Simulation Team BE LIABLE    * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *EXEMPLARY, OR CONSEQUENTIAL       * DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *PROCUREMENT OF SUBSTITUTE GOODS OR       * SERVICES; LOSS OF USE, DATA, OR
 *PROFITS; OR BUSINESS INTERRUPTION) HOWEVER       * CAUSED AND ON ANY THEORY OF
 *LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,    * OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF * THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                *
 ************************************************************************************/

#include "DecisionTree.h"
#include "Agent.h"
#include "BehaviorAttack.h"
#include "BehaviorDefense.h"
#include "BehaviorGoalie.h"
#include "BehaviorIntercept.h"
#include "BehaviorPenalty.h"
#include "BehaviorSetplay.h"
#include "Strategy.h"
#include "TimeTest.h"
#include <cstdlib>

static int decision_count = 0;
static PlayMode last_play_mode = PM_Before_Kick_Off;

bool DecisionTree::Decision(Agent &agent) {
  Assert(agent.GetSelf().IsAlive());
  fprintf(stderr, "[Decision] count=%d\n", ++decision_count);

  // RL: 检测到新的kick off，重置序列缓冲区
  PlayMode current_mode = agent.GetWorldState().GetPlayMode();
  if (current_mode == PM_Before_Kick_Off && last_play_mode != PM_Before_Kick_Off) {
    m_data_collector.ResetSequence();
    fprintf(stderr, "[DataCollector] Sequence buffer reset at kick off\n");
  }
  last_play_mode = current_mode;

  ActiveBehavior beh = Search(agent, 1);

  if (beh.GetType() != BT_None) {
    agent.SetActiveBehaviorInAct(beh.GetType());
    Assert(&beh.GetAgent() == &agent);
    bool result = beh.Execute();

    // RL: 检查带球成功并给予奖励
    m_data_collector.CheckDribbleSuccess(agent);

    // RL: 检查射门结果并给予奖励
    m_data_collector.CheckShootResult(agent);

    return result;
  }
  return false;
}

ActiveBehavior DecisionTree::Search(Agent &agent, int step) {
  if (step == 1) {
    if (agent.GetSelf().IsIdling()) {
      return ActiveBehavior(agent, BT_None);
    }

    std::list<ActiveBehavior> active_behavior_list;

    if (agent.GetSelf().IsGoalie()) {
      MutexPlan<BehaviorPenaltyPlanner>(agent, active_behavior_list) ||
          MutexPlan<BehaviorSetplayPlanner>(agent, active_behavior_list) ||
          MutexPlan<BehaviorAttackPlanner>(agent, active_behavior_list) ||
          MutexPlan<BehaviorGoaliePlanner>(agent, active_behavior_list);
    } else {
      MutexPlan<BehaviorPenaltyPlanner>(agent, active_behavior_list) ||
          MutexPlan<BehaviorSetplayPlanner>(agent, active_behavior_list) ||
          MutexPlan<BehaviorAttackPlanner>(agent, active_behavior_list) ||
          MutexPlan<BehaviorDefensePlanner>(agent, active_behavior_list);
    }

    if (!active_behavior_list.empty()) {
      fprintf(stderr, "[Search] active_behavior_list not empty, size=%zu\n", active_behavior_list.size());
      return GetBestActiveBehavior(agent, active_behavior_list);
    } else {
      return ActiveBehavior(agent, BT_None);
    }
  } else {
    return ActiveBehavior(agent, BT_None);
  }
}

/**
 * GetBestActiveBehavior - Selects the best behavior from candidates
 *
 * Selection Strategy (in order):
 *  1. Force Shoot (RL exploration): If any BT_Shoot exists, always select it
 *     - This ensures RL can learn from every shoot attempt
 *  2. NN-based Selection: If NN loaded, forward-pass all candidates, pick highest value
 *  3. Rule-based Fallback: Sort by mEvaluation heuristic, pick highest
 *
 * RL Data Collection:
 *  - All candidates added to sequence buffer regardless of selection method
 *  - On dribble/shoot selection, StartDribble/ShootTracking() is called
 *  - Later CheckDribbleSuccess/CheckShootResult() will assign rewards
 *
 * @param agent Reference to the agent
 * @param behavior_list List of candidate ActiveBehaviors from all planners
 * @return The selected ActiveBehavior to execute
 */
ActiveBehavior
DecisionTree::GetBestActiveBehavior(Agent &agent,
                                    std::list<ActiveBehavior> &behavior_list) {
  agent.SaveActiveBehaviorList(
      behavior_list); // behavior_list里面存储了本周期所有behavior决策出的最优activebehavior，这里统一保存一下，供特定behavior下周期plan时用

  // 收集所有候选行为的数据（与选择逻辑独立）
  CollectAllCandidates(agent);

  // RL: 添加所有候选到序列缓冲区（无论NN是否加载）
  // 收集数据用于训练，不管最后选哪个
  for (auto& beh : behavior_list) {
    // Build 113-dim feature vector for this candidate
    std::vector<float> features = m_data_collector.BuildFeatureVector(
        agent, beh.GetType(), beh.mTarget, beh.mPower);

    float nn_value = 0.0f;
    if (use_nn_ && nn_.Loaded()) {
      // NN inference
      nn_value = nn_.Forward(features);
    }

    // RL: 添加到序列缓冲区
    m_data_collector.AddToSequence(agent, beh.GetType(), beh.mTarget, beh.mPower, nn_value);
  }

  fprintf(stderr, "[Decision] unum=%d Before NN check: use_nn_=%d\n", agent.GetSelf().GetUnum(), use_nn_ ? 1 : 0);
    fflush(stderr);

    // ========== Step 1: RL Exploration - Force Shoot ==========
    // 如果有shoot候选，100%强制选择shoot（用于学习射门RL）
    // 这个逻辑在NN选择之前执行，确保射门能被选中
    for (auto& beh : behavior_list) {
      if (beh.GetType() == BT_Shoot) {
        fprintf(stderr, "[Decision] Force shoot selected! eval=%.2f\n", beh.mEvaluation);
        m_data_collector.StartShootTracking(agent, beh.mTarget);
        return beh;
      }
    }

    // ========== Step 2: NN-based Selection ==========
    if (use_nn_ && nn_.Loaded()) {
    float best_value = -1e9f;
    ActiveBehavior* best = nullptr;

    for (auto& beh : behavior_list) {
      // Build 113-dim feature vector for this candidate
      std::vector<float> features = m_data_collector.BuildFeatureVector(
          agent, beh.GetType(), beh.mTarget, beh.mPower);

      // NN inference
      float nn_value = nn_.Forward(features);

      if (nn_value > best_value) {
        best_value = nn_value;
        best = &beh;
      }
    }

    if (best != nullptr) {
      // RL: 如果选中带球或射门，开始追踪
      if (best->GetType() == BT_Dribble) {
        m_data_collector.StartDribbleTracking(agent);
      } else if (best->GetType() == BT_Shoot) {
        m_data_collector.StartShootTracking(agent, best->mTarget);
      }
      return *best;
    }
  }

  // ========== Step 3: Rule-based Fallback ==========
  // NN not loaded or failed - use heuristic evaluation
  behavior_list.sort(std::greater<ActiveBehavior>());

  // RL: 如果选中带球或射门，开始追踪
  if (behavior_list.front().GetType() == BT_Dribble) {
    m_data_collector.StartDribbleTracking(agent);
  } else if (behavior_list.front().GetType() == BT_Shoot) {
    m_data_collector.StartShootTracking(agent, behavior_list.front().mTarget);
  }

  return behavior_list.front();
}

void DecisionTree::CollectAllCandidates(Agent &agent) {
  // 直接调用各个Planner生成所有候选
  // 原来用于收集数据，现在仅保留结构用于RL序列收集
  // 注意：原有的 Collect() 调用已禁用，现在是增量学习模式

  if (agent.GetSelf().IsIdling()) {
    return;
  }

  std::list<ActiveBehavior> all_candidates;

  if (agent.GetSelf().IsGoalie()) {
    MutexPlan<BehaviorPenaltyPlanner>(agent, all_candidates);
    MutexPlan<BehaviorSetplayPlanner>(agent, all_candidates);
    MutexPlan<BehaviorAttackPlanner>(agent, all_candidates);
    MutexPlan<BehaviorGoaliePlanner>(agent, all_candidates);
  } else {
    MutexPlan<BehaviorPenaltyPlanner>(agent, all_candidates);
    MutexPlan<BehaviorSetplayPlanner>(agent, all_candidates);
    MutexPlan<BehaviorAttackPlanner>(agent, all_candidates);
    MutexPlan<BehaviorDefensePlanner>(agent, all_candidates);
  }

  // 原有数据收集已禁用，现在只通过 GetBestActiveBehavior 中的 AddToSequence 收集RL数据
}
