# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WrightEagleBASE is a RoboCup Soccer Simulation 2D team framework implementing the MAXQ-OP hierarchical planning algorithm. The codebase is in C++ with Python training scripts for neural network value inference.

## Build Commands

```bash
make debug       # Build debug version (with assertions and debug info)
make release     # Build release version
make clean       # Clean all build artifacts
make             # Default builds debug
```

Binary outputs: `Debug/WEBase` and `Release/WEBase`

## Running

Requires RoboCup Soccer 2D Simulator (rcssserver, rcssmonitor):
```bash
./start.sh                    # Start team "WEBase" on left side
./start.sh -t [TEAMNAME]      # Start team with custom name on right side
./start.sh -v Release         # Use release build
```

NN model loading: Set `NN_MODEL` environment variable to path of `.bin` model file.

## Architecture

### Main Loop Flow
1. `Player::Run()` in `src/Player.cpp` - sensing, decision-making, execution cycle
2. `DecisionTree::Decision()` - main decision entry point
3. `DecisionTree::Search()` - recursive search through behavior hierarchy
4. `ActiveBehavior::Execute()` - executes chosen behavior

### Key Components

- **Agent** (`src/Agent.h`): Represents a player/coach, provides action interfaces (Turn, Dash, Kick, etc.)
- **WorldState** (`src/WorldState.h`): Game state (ball, players, positions)
- **DecisionTree** (`src/DecisionTree.h`): Behavior selection hierarchy with MAXQ decomposition
- **Behaviors** (`src/Behavior*.cpp`): Individual behavior implementations following Plan/Execute pattern
- **Formation** (`src/Formation.h`): Team formation and positioning

### Behavior System
Behaviors use a Factory pattern with `Plan()` (decision-making) and `Execute()` (action execution):
- Attack behaviors: Dribble, Pass, Shoot, Hold, Intercept
- Defense behaviors: Block, Mark, Position
- Goalie, Formation, Penalty

### Neural Network Integration
- `src/NNValueInference.h`: Loads binary NN models for value estimation
- `src/DataCollector.h`: Collects training data during games
- `train/train_value.py`: PyTorch training script, exports `.bin` models for C++

### State Update Order (in `Player::Run()`)
1. Formation updates
2. Communication system (hear parsing)
3. World state update
4. Strategy update
5. Decision tree execution
6. Command sending

## Key File Locations

- Source: `src/`
- Debug build: `Debug/`
- Release build: `Release/`
- Log files: `Logfiles/`
- Trained models: `models/`
- Training scripts: `train/`
