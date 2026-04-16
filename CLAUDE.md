# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WrightEagleBASE is a RoboCup Soccer Simulation 2D team framework implementing the MAXQ-OP hierarchical planning algorithm. The codebase is in C++ with Python training scripts for neural network value inference.

## Build Commands

```bash
make             # Build debug version (default)
make debug       # Build debug version (with assertions and debug info)
make release     # Build release version
make clean       # Clean all build artifacts
```

Binary outputs: `Debug/WEBase` and `Release/WEBase`

## Running

Requires RoboCup Soccer 2D Simulator (rcssserver, rcssmonitor):
```bash
./start.sh                    # Start team "WEBase" on left side
./start.sh -t [TEAMNAME]      # Start team with custom name on right side
./start.sh -v Release         # Use release build
```

### Neural Network Integration
```bash
NN_MODEL=/path/to/model.bin   # Path to binary NN model file
USE_NN=1                      # Enable NN value inference (optional, auto-enabled if NN_MODEL set)
```
When `NN_MODEL` is set or `USE_NN=1`, `DecisionTree` uses `NNValueInference` to evaluate behavior candidates.

## Architecture

### Main Loop Flow
1. `Player::Run()` in `src/Player.cpp` - sensing, decision-making, execution cycle
2. `DecisionTree::Decision()` - main decision entry point
3. `DecisionTree::Search()` - recursive search through behavior hierarchy
4. `ActiveBehavior::Execute()` - executes chosen behavior

### State Update Order (in `Player::Run()`)
1. Formation updates
2. Communication system (hear parsing)
3. World state update
4. Strategy update
5. Decision tree execution
6. Command sending

### Key Components

| Component | File | Role |
|-----------|------|------|
| **Agent** | `src/Agent.h` | Player/coach interface; actions: Turn, Dash, Kick, etc. |
| **WorldState** | `src/WorldState.h` | Game state: ball, players, positions |
| **DecisionTree** | `src/DecisionTree.h` | Behavior selection hierarchy with MAXQ decomposition |
| **NNValueInference** | `src/NNValueInference.h` | Loads binary NN models for value estimation (113→64→32→16→1) |
| **DataCollector** | `src/DataCollector.h` | Collects training data during games |

### Behavior System (MAXQ-OP)

Each behavior has two parts:
- **Planner** (`Behavior*Planner`): Evaluates subtasks via `Plan()`, produces `ActiveBehavior` candidates
- **Executer** (`Behavior*Executer`): Runs `Execute()` to perform robot actions

The `BehaviorFactory` auto-registers behaviors. All behaviors inherit from:
- `BehaviorPlannerBase<BehaviorDataType>` for planners
- `BehaviorExecuterBase<BehaviorDataType>` for executers

**Available behaviors:**
- Attack: `Dribble`, `Hold`, `Pass`, `Shoot`, `Intercept`
- Defense: `Block`, `Mark`, `Position`
- Other: `Goalie`, `Formation`, `Penalty`, `Setplay`

### Modification Entry Points

To modify team behavior, focus on `Behavior*::Plan()` functions in `src/Behavior*.cpp`:
- `Plan()` evaluates and selects subtasks (where behavioral decisions are made)
- `Execute()` performs the actual robot actions

### Neural Network Training

```bash
python train/train_value.py --data_file <path> --output_prefix <name>
python train/do_train_incremental.py  # Incremental training
```

Output: `.bin` model files loaded by `NNValueInference`.
NN model structure: 113 inputs → 64 → 32 → 16 → 1 (value output).

## Key Files

```
src/
  Player.cpp            # Main loop (Run())
  DecisionTree.cpp/h    # Decision hierarchy (Search(), Decision())
  BehaviorBase.h        # Planner/Executer base classes, ActiveBehavior, BehaviorFactory
  BehaviorAttack.h      # Attack behavior planners
  BehaviorDefense.h     # Defense behavior planners
  BehaviorDribble.cpp   # Dribble implementation (Plan + Execute)
  BehaviorShoot.cpp    # Shoot implementation
  BehaviorPass.cpp     # Pass implementation
  NNValueInference.cpp/h # NN model loading and inference
  DataCollector.cpp/h  # Training data collection
  Agent.h               # Action interface (Turn, Dash, Kick, etc.)
  WorldState.h          # Game state representation
  Formation.h           # Team formation and positioning
  Types.h               # Enums: PlayMode, BehaviorType, Situation, etc.

train/
  train_value.py        # PyTorch training script
  do_train_incremental.py # Incremental training

Debug/WEBase, Release/WEBase  # Compiled binaries
models/                       # Trained NN model files
Logfiles/                     # Game logs
```

## Important Enums (from `Types.h`)

- `PlayMode`: `PM_Play_On`, `PM_Our_Kick_Off`, `PM_Opp_Kick_Off`, `PM_Before_Kick_Off`, etc.
- `BehaviorType`: `BT_Dribble`, `BT_Shoot`, `BT_Pass`, `BT_Block`, `BT_Mark`, etc.
- `Situation`: `ST_Forward_Attack`, `ST_Penalty_Attack`, `ST_Defense`

---

## Neural Network Feature Vector (113 dimensions)

The NN uses a 113-dimensional feature vector combining global state + candidate action:

### Global Features (109 dims, indices 0-108)

| Index | Feature | Dim | Description |
|-------|---------|-----|-------------|
| 0-1 | Ball Position X/Y | 2 | Ball center position |
| 2-3 | Ball Velocity X/Y | 2 | Ball velocity vector |
| 4-5 | Self Position X/Y | 2 | This player position |
| 6-7 | Self Velocity X/Y | 2 | This player velocity |
| 8 | Self Body Direction | 1 | Body angle in degrees |
| 9 | Self Is Goalie | 1 | 1.0 if goalie, else 0.0 |
| 10 | Ball Is Kickable | 1 | Self to ball < kickable_area |
| 11-32 | Teammate 1-11 Positions X/Y | 22 | Each teammate (99.0 if dead) |
| 33-54 | Teammate 1-11 Velocities X/Y | 22 | Each teammate velocity (0 if dead) |
| 55-76 | Opponent 1-11 Positions X/Y | 22 | Each opponent (99.0 if dead) |
| 77-98 | Opponent 1-11 Velocities X/Y | 22 | Each opponent velocity (0 if dead) |
| 99 | Play Mode | 1 | Current play mode enum value |
| 100 | Ball Ownership | 1 | Reserved (always 0.0) |
| 101-102 | Left Goalie Position X/Y | 2 | Left team goalkeeper |
| 103-104 | Left Goalie Velocity X/Y | 2 | Left goalkeeper velocity |
| 105-106 | Right Goalie Position X/Y | 2 | Right team goalkeeper |
| 107-108 | Right Goalie Velocity X/Y | 2 | Right goalkeeper velocity |

### Candidate Features (4 dims, indices 109-112)

| Index | Feature | Description |
|-------|---------|-------------|
| 109 | Behavior Type | Enum value of the candidate action |
| 110-111 | Target X/Y | Action target position |
| 112 | Power | Action power parameter |

### RL Reward Signals

**Dribble**: After 15 cycles, check if still near ball
- Success (still kickable): reward = +1.0
- Failure (lost ball): reward = -1.0

**Shoot**: After 25 cycles, check play mode
- Goal (PM_Goal_Ours): reward = +1.0
- Miss (no goal): reward = -1.0

---

## Decision Flow with Exploration Strategy

```
GetBestActiveBehavior()
  │
  ├─► 1. Force Shoot Check (RL exploration)
  │     If any BT_Shoot candidate exists → always select it
  │     (This ensures RL learns from shoot outcomes)
  │
  ├─► 2. NN-based Selection (if NN loaded)
  │     For each candidate:
  │       - Build 113-dim feature vector
  │       - Forward pass → get value estimate
  │       - Select candidate with highest NN value
  │
  └─► 3. Rule-based Fallback (if NN not loaded)
        - Sort by mEvaluation (heuristic)
        - Select highest evaluation
```

**RL Exploration**: Force-shoot strategy ensures every shoot attempt is collected for training, even if NN would not select it.

---

## RL Training Pipeline

### Data Collection (during gameplay)
```
1. DecisionTree::GetBestActiveBehavior()
   - Builds 113-dim features for ALL candidates
   - Adds each to DataCollector sequence buffer

2. When dribble selected → StartDribbleTracking()
   - Records start cycle, ball position

3. When shoot selected → StartShootTracking()
   - Records start cycle, target, ball position

4. After behavior executes:
   - CheckDribbleSuccess() → if 15 cycles elapsed
   - CheckShootResult() → if 25 cycles elapsed

5. On reward event:
   - RewardSequence() → TD target calculation
   - γ = 0.9 (discount factor)
   - TD target = r + γ * V(s')
```

### Data Format (JSONL)
```json
{"f":[...113 floats...],"v":0.85}
```

### Training
```bash
python train/train_value.py --data_file <path> --output_prefix <name>
```

### Deployment
```bash
NN_MODEL=/path/to/model.bin ./start.sh
```

---

## Key Design Patterns

### 1. Factory with Auto-Registration
```cpp
// Each Behavior*Executer self-registers at static init time
namespace {
bool ret = BehaviorExecutable::AutoRegister<BehaviorDribbleExecuter>();
}
```
**Why**: Eliminates manual registration; adding a new behavior only requires creating the class.

### 2. Planner/Executer Separation
- **Planner** (`Behavior*Planner`): Inherits `BehaviorPlannerBase<DataType>`. `Plan()` generates candidates with `mEvaluation` scores.
- **Executer** (`Behavior*Executer`): Inherits `BehaviorExecuterBase<DataType>`. `Execute()` performs actions.

### 3. Data Lazy Initialization
```cpp
template <class DecisionDataDerived>
DecisionDataDerived& Agent::GetDecisionData(DecisionData** ptr) {
    if (*ptr == nullptr) { *ptr = new DecisionDataDerived(*this); }
    (*ptr)->Update();
    return **ptr;
}
```
**Why**: Avoids constructing expensive data structures until first access.

### 4. Singleton Pattern
Used for: `BehaviorFactory`, `Evaluation`, `Formation`, `Logger`, `ServerParam`, `PlayerParam`

---

## Behavior Modification Guide

### To Modify Existing Behavior Evaluation (e.g., make dribble more aggressive):

1. Find `BehaviorDribblePlanner::Plan()` in `src/BehaviorDribble.cpp`
2. Look for where `mEvaluation` is calculated
3. Adjust weights or add new terms to the evaluation formula
4. Rebuild: `make`

### To Add a New Behavior (e.g., `BT_Custom`):

1. Add `BT_Custom` to `BehaviorType` enum in `Types.h`
2. Create `BehaviorCustomPlanner` inheriting `BehaviorPlannerBase<BehaviorAttackData>`
3. Create `BehaviorCustomExecuter` inheriting `BehaviorExecuterBase<BehaviorAttackData>`
4. In `.cpp`, add the auto-registration pattern:
   ```cpp
   const BehaviorType BehaviorCustomExecuter::BEHAVIOR_TYPE = BT_Custom;
   namespace { bool ret = BehaviorExecutable::AutoRegister<BehaviorCustomExecuter>(); }
   ```
5. Implement `Plan()` to generate candidates with `mEvaluation`
6. Implement `Execute()` to perform actions via `mpAgent->Dash()`, `mpAgent->Kick()`, etc.
7. Register planner in `DecisionTree::Search()` if needed

---

## Common Tasks

### Debug NN Inference
```bash
USE_NN=1 NN_MODEL=models/your_model.bin ./start.sh 2>&1 | grep -E "(NN|Decision)"
```

### Collect RL Training Data
```bash
# Run games and data auto-saves to Logfiles/rl_data_incremental.json
./start.sh
```

### Train NN Model
```bash
python train/train_value.py --data_file Logfiles/rl_data.json --output_prefix models/v1
```

### Incremental Training (continue from existing model)
```bash
python train/do_train_incremental.py --base_model models/v1.bin --data_file Logfiles/new_data.json
```
