# Reinforcement Learning Algorithms: Mathematical Overview

This document provides a concise mathematical explanation of key reinforcement learning algorithms, organized by environment types.

## Core RL Concepts

### Markov Decision Process (MDP)
State transition probability:
```
P(s', r|s, a) = P{St = s', Rt = r | St-1 = s, At-1 = a}
```

### Discounted Reward
```
Gt = E[∑(k=0 to ∞) γ^k * Rt+k+1]
```
where γ ∈ [0,1] is the discount factor.

### Action-Value Function
```
Qπ(s,a) = Eπ[∑(k=0 to ∞) γ^k * Rt+k+1 | St = s, At = a]
```

### Bellman Optimality Equation
```
Q*(s,a) = E[Rt+1 + γ max(a') Q*(s',a')]
```

---

## Category 1: Limited States & Discrete Actions

### Q-Learning
**Purpose**: Find optimal policy in simple environments with finite state-action spaces.

**Update Rule**:
```
Q(s,a) ← Q(s,a) + α[r + γ max(a') Q(s',a') - Q(s,a)]
```

**Key Features**:
- Maintains Q-table for all state-action pairs
- Off-policy algorithm
- Converges to optimal Q-values

### SARSA (State-Action-Reward-State-Action)
**Purpose**: Similar to Q-learning but uses on-policy updates.

**Update Rule**:
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

**Key Difference**: Uses the actual next action a' taken by the policy, not the max.

---

## Category 2: Unlimited States & Discrete Actions

### Deep Q-Networks (DQN)
**Purpose**: Extend Q-learning to high-dimensional state spaces using deep neural networks.

**Loss Function**:
```
L(θ) = E[(Target - Q(s,a;θ))²]
Target = E[Rt+1 + γ max(a') Q*(s',a')]
```

**Key Innovation**: Uses experience replay and target networks.

### Double DQN
**Purpose**: Address overestimation bias in DQN.

**Target Calculation**:
```
Target = r + γ Q(s', argmax(a') Q(s',a';θ); θ-)
```
Uses online network to select action, target network to evaluate.

### Dueling DQN
**Purpose**: Better value estimation by separating state value and advantage.

**Architecture**:
```
Q(s,a) = V(s) + A(s,a) - (1/|A|)∑(a') A(s,a')
```
where:
- V(s): State-value function
- A(s,a): Advantage function

### Deep Recurrent Q-Networks (DRQN)
**Purpose**: Handle partially observable environments using LSTM.

**Key Feature**: Replaces fully connected layer with LSTM to integrate temporal information.

---

## Category 3: Unlimited States & Continuous Actions

### Policy Gradient Methods

**Objective Function**:
```
J(πθ) = ∑s ρπθ(s) ∑a Qπθ(s,a) πθ(a|s)
```

**Policy Gradient Theorem**:
```
∇θJ(θ) = Es~ρπθ, a~πθ[Qπθ(s,a) ∇θ ln πθ(at|st)]
```

**Parameter Update**:
```
θt+1 = θt + α∇J(θt)
```

### REINFORCE
**Purpose**: Monte Carlo policy gradient method.

**Update Rule**:
```
∇θJ(θ) = Eπ[Gt ∇θ ln πθ(At|St)]
```

**Key Features**:
- Uses complete episodes
- High variance, unbiased estimates
- Can add baseline to reduce variance

### Trust Region Policy Optimization (TRPO)
**Purpose**: Constrained policy updates for stable learning.

**Constraint**: KL-divergence between old and new policies stays within trust region.

### Proximal Policy Optimization (PPO)
**Purpose**: Simplified version of TRPO with clipped objective.

**Key Feature**: Clips policy ratio to stay within acceptable range.

---

## Actor-Critic Methods

### Basic Actor-Critic
**Components**:
- **Actor**: Updates policy parameters θ using policy gradients
- **Critic**: Estimates value function Qw(s,a) with parameters w

### Deep Deterministic Policy Gradient (DDPG)
**Purpose**: Continuous control with deterministic policies.

**Deterministic Policy Gradient**:
```
∇θJ(μθ) = Es~ρβ[∇θ μθ(s) ∇a Q^μ(s,a)|a=μθ(s)]
```

**Key Features**:
- Deterministic policy μ(s) instead of stochastic π(s,a)
- Uses target networks and experience replay
- Off-policy algorithm

### Twin Delayed DDPG (TD3)
**Purpose**: Address overestimation in DDPG.

**Key Improvements**:
- Uses two critic networks, takes minimum
- Delays policy updates
- Adds noise to target actions

### Asynchronous Advantage Actor-Critic (A3C)
**Purpose**: Parallel training with multiple agents.

**Key Features**:
- Multiple workers train asynchronously
- Each maintains local policy πθ(at|st) and value function Vθ(st)
- Updates global network parameters

### Soft Actor-Critic (SAC)
**Purpose**: Maximum entropy reinforcement learning.

**Objective**:
```
J(π) = ∑t Es~ρπ, a~π[r(st,at) + αH(π(·|st))]
```

**Key Feature**: Maximizes both reward and entropy for exploration.

### IMPALA (Importance Weighted Actor-Learner Architecture)
**Purpose**: Scalable distributed RL with off-policy correction.

**Key Features**:
- Decouples acting from learning
- Uses V-trace for off-policy correction
- Supports single or multiple learners

---

## Algorithm Selection Guide

| Environment Type | States | Actions | Recommended Algorithms |
|-----------------|---------|---------|----------------------|
| Simple | Limited | Discrete | Q-Learning, SARSA |
| Complex | Unlimited | Discrete | DQN, Double DQN, Dueling DQN |
| Continuous Control | Unlimited | Continuous | PPO, SAC, DDPG, TD3 |

---

## Key Mathematical Relationships

### Off-Policy vs On-Policy
- **On-Policy**: Training data follows target policy πθ
- **Off-Policy**: Training data follows behavior policy β, corrected by importance sampling ratio πθ(a|s)/β(a|s)

### Variance-Bias Trade-off
- **Monte Carlo**: Unbiased, high variance
- **Temporal Difference**: Biased, low variance
- **Actor-Critic**: Combines both approaches

### Experience Replay
Experience tuple: e(s, a, s', r)
- Uniform sampling: Random selection from replay buffer
- Prioritized replay: Sample based on TD-error magnitude
