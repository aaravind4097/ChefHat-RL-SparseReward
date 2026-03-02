# PPO with Curriculum Learning for Chef's Hat (Variant 2)

**Student ID:** 16779086

## 1. Overview

This project implements a Proximal Policy Optimization (PPO) agent for the game *Chef's Hat*, focusing on **Variant 2: Curriculum Learning**. The agent is trained progressively against increasingly difficult opponents, starting with `RandomAgent` and graduating to `HeuristicAgentV1` and `HeuristicAgentV2`.

### Key Features:
- **Curriculum Manager:** A dedicated module to manage the training stages and promotion criteria.
- **PPO Agent:** A standard PPO implementation with action masking.
- **Staged Training:** The training script automatically handles the transition between curriculum stages based on win-rate thresholds.

## 2. Project Structure

```
/task2_project_16779086
├── configs/
│   └── config.yaml         # Curriculum stages and hyperparameters
├── curriculum/
│   └── manager.py          # Manages training stages
├── ppo/
│   ├── model.py            # Actor-Critic networks
│   └── ppo_agent.py        # PPO agent logic
├── training/
│   └── trainer.py          # Curriculum-based training loop
...
```

## 3. Usage

### 3.1. Training

Run the main training script. The curriculum manager will handle the rest.

```bash
python3 run_train.py
```

### 3.2. Evaluation

Evaluate the final trained agent against all opponent types.

```bash
python3 run_eval.py
```
