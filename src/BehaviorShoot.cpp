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

#include "BehaviorShoot.h"
#include "ActionEffector.h"
#include "BehaviorBase.h"
#include "Dasher.h"
#include "Evaluation.h"
#include "Geometry.h"
#include "InterceptModel.h"
#include "Kicker.h"
#include "PlayerParam.h"
#include "PositionInfo.h"
#include "ServerParam.h"
#include "Tackler.h"
#include "TimeTest.h"
#include "Utilities.h"
#include "VisualSystem.h"
#include <cmath>
#include <list>

using namespace std;

namespace {
bool ret = BehaviorExecutable::AutoRegister<BehaviorShootExecuter>();
}

const BehaviorType BehaviorShootExecuter::BEHAVIOR_TYPE = BT_Shoot;

/**
 * BehaviorShootExecuter's constructor
 */
BehaviorShootExecuter::BehaviorShootExecuter(Agent &agent)
    : BehaviorExecuterBase<BehaviorAttackData>(agent) {
  Assert(ret);
}

/**
 * BehaviorShootExecuter's constructor
 */
BehaviorShootExecuter::~BehaviorShootExecuter() {}

/**
 * Execute an active behavior as shoot. May raise ball or player.
 * \param act_bhv
 * \return true  if success;
 *         false otherwise.
 */
bool BehaviorShootExecuter::Execute(const ActiveBehavior &shoot) {
  Logger::instance().LogShoot(mBallState.GetPos(), shoot.mTarget, "@Shoot");
  if (shoot.mDetailType == BDT_Shoot_Tackle)
    return Tackler::instance().TackleToDir(mAgent, shoot.mAngle);
  else
    return Kicker::instance().KickBall(mAgent, shoot.mTarget,
                                       ServerParam::instance().ballSpeedMax(),
                                       KM_Quick, 0, true);
}

/**
 * BehaviorShootPlanner's constructor
 */
BehaviorShootPlanner::BehaviorShootPlanner(Agent &agent)
    : BehaviorPlannerBase<BehaviorAttackData>(agent) {}

/**
 * BehaviorShootPlanner's destructor
 */
BehaviorShootPlanner::~BehaviorShootPlanner() {}

/**
 * Plan.
 * None or one ActiveBehavior will be push back to behavior_list.
 */
void BehaviorShootPlanner::Plan(list<ActiveBehavior> &behavior_list) {
  if (!mSelfState.IsKickable())
    return;

  if (mWorldState.GetPlayMode() == PM_Our_Foul_Charge_Kick ||
      mWorldState.GetPlayMode() == PM_Our_Back_Pass_Kick ||
      mWorldState.GetPlayMode() == PM_Our_Indirect_Free_Kick ||
      (mWorldState.GetLastPlayMode() == PM_Our_Indirect_Free_Kick &&
       mAgent.IsLastActiveBehaviorInActOf(
           BT_Pass))) // Indircet后传球保持不射门，至少跑位后上个动作会改
  {
    return;
  }

  if (mSelfState.GetPos().X() >
      ServerParam::instance().pitchRectanglar().Right() -
          PlayerParam::instance().shootMaxDistance()) {
    // 球门柱往里收一点，提高射门准确率
    const double kInwardOffset = 0.5;  // 向球门内侧偏移0.5米
    Vector leftPost = ServerParam::instance().oppLeftGoalPost();
    Vector rightPost = ServerParam::instance().oppRightGoalPost();
    // 左边门柱往下移，右边门柱往上移（都向球门中心靠近）
    leftPost.SetY(leftPost.Y() > 0 ? leftPost.Y() - kInwardOffset : leftPost.Y() + kInwardOffset);
    rightPost.SetY(rightPost.Y() > 0 ? rightPost.Y() - kInwardOffset : rightPost.Y() + kInwardOffset);

    AngleDeg left =
        (leftPost - mSelfState.GetPos()).Dir();
    AngleDeg right =
        (rightPost - mSelfState.GetPos())
            .Dir();
    Vector target;
    AngleDeg interval;
    Line c(leftPost, rightPost);
    AngleDeg shootDir =
        mPositionInfo.GetShootAngle(left, right, mSelfState, interval);
    if (interval < mSelfState.GetRandAngle(
                       ServerParam::instance().maxPower(),
                       ServerParam::instance().ballSpeedMax(), mBallState) *
                       3) {
      return;
    }

    Ray f(mSelfState.GetPos(), shootDir);
    c.Intersection(f, target);

    // 生成3个射门候选: 左门柱、中间、右门柱
    Vector shootTargets[3] = {leftPost, target, rightPost};
    AngleDeg shootDirs[3] = {left, shootDir, right};

    for (int i = 0; i < 3; ++i) {
      if (Tackler::instance().CanTackleToDir(mAgent, shootDirs[i]) &&
          Tackler::instance().GetBallVelAfterTackle(mAgent, shootDirs[i]).Mod() >
              ServerParam::instance().ballSpeedMax() - 0.05) {
        ActiveBehavior shoot(mAgent, BT_Shoot, BDT_Shoot_Tackle);
        shoot.mTarget = shootTargets[i];
        shoot.mAngle = shootDirs[i];
        shoot.mEvaluation = 2.0 + FLOAT_EPS;
        mActiveBehaviorList.push_back(shoot);
      } else {
        ActiveBehavior shoot(mAgent, BT_Shoot);
        shoot.mTarget = shootTargets[i];
        shoot.mEvaluation = 2.0 + FLOAT_EPS;
        mActiveBehaviorList.push_back(shoot);
      }
    }
  }
}
