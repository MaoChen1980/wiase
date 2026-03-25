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
bool DecisionTree::Decision(Agent &agent) {
  Assert(agent.GetSelf().IsAlive());
  fprintf(stderr, "[Decision] count=%d\n", ++decision_count);

  ActiveBehavior beh = Search(agent, 1);

  if (beh.GetType() != BT_None) {
    agent.SetActiveBehaviorInAct(beh.GetType());
    Assert(&beh.GetAgent() == &agent);
    bool result = beh.Execute();

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
      return GetBestActiveBehavior(agent, active_behavior_list);
    } else {
      return ActiveBehavior(agent, BT_None);
    }
  } else {
    return ActiveBehavior(agent, BT_None);
  }
}

ActiveBehavior
DecisionTree::GetBestActiveBehavior(Agent &agent,
                                    std::list<ActiveBehavior> &behavior_list) {
  agent.SaveActiveBehaviorList(
      behavior_list); // behavior_list里面存储了本周期所有behavior决策出的最优activebehavior，这里统一保存一下，供特定behavior下周期plan时用

  // 收集所有候选行为的数据（与选择逻辑独立）
  CollectAllCandidates(agent);

  if (use_nn_ && nn_.Loaded()) {
    // NN-based selection
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
      return *best;
    }
  }

  // Rule-based selection (existing)
  behavior_list.sort(std::greater<ActiveBehavior>());
  return behavior_list.front();
}

void DecisionTree::CollectAllCandidates(Agent &agent) {
  // 直接调用各个Planner生成所有候选，收集数据
  // 这样与原来的behavior_list选择逻辑独立

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

  // 为每个候选收集数据
  for (auto& beh : all_candidates) {
    if (beh.GetType() != BT_None && beh.mEvaluation > 0) {
      m_data_collector.Collect(agent, beh.GetType(), beh.mTarget,
                               beh.mPower, beh.mEvaluation);
    }
  }
}
