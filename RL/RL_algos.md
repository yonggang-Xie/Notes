# Reinforcement Learning Algorithms: Mathematical Overview

This document provides a concise mathematical explanation of key reinforcement learning algorithms, organized by environment types.

## Core RL Concepts

### Markov Decision Process (MDP)
State transition probability:
```
P(s', r|s, a) = P{St = s', Rt = r | St-1 = s, At-1 = a}
```
**Terms**:
- `s`: Current state
- `s'`: Next state
- `a`: Action taken
- `r`: Reward received

**Rationale**: Defines the probability of transitioning to state `s'` and receiving reward `r` when taking action `a` in state `s`. The Markov property assumes the future depends only on the current state, not the history.

### Discounted Reward
```
Gt = E[∑(k=0 to ∞) γ^k * Rt+k+1]
```
**Terms**:
- `Gt`: Expected return from time t
- `γ`: Discount factor (0 ≤ γ ≤ 1)
- `Rt+k+1`: Reward at time t+k+1
- `E[·]`: Expected value

**Rationale**: Calculates the total expected reward from time t onwards. The discount factor γ makes immediate rewards more valuable than future rewards, ensuring convergence and reflecting uncertainty about the future.

### Action-Value Function
```
Qπ(s,a) = Eπ[∑(k=0 to ∞) γ^k * Rt+k+1 | St = s, At = a]
```
**Terms**:
- `Qπ(s,a)`: Value of taking action `a` in state `s` under policy π
- `π`: Policy (strategy for selecting actions)
- `|`: Conditional on starting in state s and taking action a

**Rationale**: Estimates the expected cumulative reward when starting in state `s`, taking action `a`, and following policy π thereafter. This helps evaluate the quality of actions.

### Bellman Optimality Equation
```
Q*(s,a) = E[Rt+1 + γ max(a') Q*(s',a')]
```
**Terms**:
- `Q*(s,a)`: Optimal action-value function
- `Rt+1`: Immediate reward
- `max(a') Q*(s',a')`: Best possible value from next state
- `s'`: Next state after taking action a

**Rationale**: The optimal value of an action equals the immediate reward plus the discounted value of the best action in the next state. This recursive relationship is fundamental to dynamic programming in RL.

---

## Category 1: Limited States & Discrete Actions

### Q-Learning
**Purpose**: Find optimal policy in simple environments with finite state-action spaces.

**Update Rule**:
```
Q(s,a) ← Q(s,a) + α[r + γ max(a') Q(s',a') - Q(s,a)]
```
**Terms**:
- `α`: Learning rate (0 < α ≤ 1)
- `r`: Observed reward
- `max(a') Q(s',a')`: Greedy action selection in next state
- `[r + γ max(a') Q(s',a') - Q(s,a)]`: TD-error (temporal difference error)

**Rationale**: Updates Q-values toward the Bellman optimality equation. The TD-error measures the difference between current estimate and target value. Uses off-policy learning (learns optimal policy while following exploratory behavior).

### SARSA (State-Action-Reward-State-Action)
**Purpose**: Similar to Q-learning but uses on-policy updates.

**Update Rule**:
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```
**Terms**:
- `a'`: Actual next action taken by current policy (not max)
- Other terms same as Q-learning

**Rationale**: Updates Q-values based on the actual action taken by the current policy, making it on-policy. More conservative than Q-learning since it doesn't assume optimal future actions.

---

## Category 2: Unlimited States & Discrete Actions

### Deep Q-Networks (DQN)
**Purpose**: Extend Q-learning to high-dimensional state spaces using deep neural networks.

**Loss Function**:
```
L(θ) = E[(Target - Q(s,a;θ))²]
Target = E[Rt+1 + γ max(a') Q*(s',a')]
```
**Terms**:
- `θ`: Neural network parameters
- `Q(s,a;θ)`: Q-value predicted by network
- `Target`: Target Q-value from Bellman equation
- `L(θ)`: Mean squared error loss

**Rationale**: Trains neural network to approximate Q-function by minimizing squared difference between predicted and target Q-values. Uses function approximation to handle large state spaces where tabular methods fail.

### Double DQN
**Purpose**: Address overestimation bias in DQN.

**Target Calculation**:
```
Target = r + γ Q(s', argmax(a') Q(s',a';θ); θ-)
```
**Terms**:
- `θ`: Online network parameters
- `θ-`: Target network parameters (periodically updated)
- `argmax(a') Q(s',a';θ)`: Action selection using online network
- `Q(s', ·; θ-)`: Action evaluation using target network

**Rationale**: Separates action selection from action evaluation to reduce overestimation bias. Online network selects actions, target network evaluates them, preventing the same network from both selecting and overestimating actions.

### Dueling DQN
**Purpose**: Better value estimation by separating state value and advantage.

**Architecture**:
```
Q(s,a) = V(s) + A(s,a) - (1/|A|)∑(a') A(s,a')
```
**Terms**:
- `V(s)`: State-value function (value of being in state s)
- `A(s,a)`: Advantage function (how much better action a is than average)
- `|A|`: Number of possible actions
- `(1/|A|)∑(a') A(s,a')`: Mean advantage (for identifiability)

**Rationale**: Explicitly separates the value of being in a state from the advantage of taking specific actions. The mean subtraction ensures unique decomposition and improves learning stability.

### Deep Recurrent Q-Networks (DRQN)
**Purpose**: Handle partially observable environments using LSTM.

**Key Feature**: Replaces fully connected layer with LSTM to integrate temporal information across multiple time steps.

**Rationale**: LSTM memory allows the network to remember past observations, crucial for environments where current state doesn't contain complete information.

---

## Category 3: Unlimited States & Continuous Actions

### Policy Gradient Methods

**Objective Function**:
```
J(πθ) = ∑s ρπθ(s) ∑a Qπθ(s,a) πθ(a|s)
```
**Terms**:
- `πθ(a|s)`: Parameterized policy (probability of action a in state s)
- `ρπθ(s)`: State visitation distribution under policy πθ
- `Qπθ(s,a)`: Action-value function under policy πθ

**Rationale**: Defines the expected performance of a parameterized policy. Maximizing this objective leads to better policies by increasing the probability of good actions.

**Policy Gradient Theorem**:
```
∇θJ(θ) = Es~ρπθ, a~πθ[Qπθ(s,a) ∇θ ln πθ(at|st)]
```
**Terms**:
- `∇θJ(θ)`: Gradient of objective with respect to parameters
- `∇θ ln πθ(at|st)`: Score function (gradient of log probability)
- `Es~ρπθ, a~πθ[·]`: Expectation over state-action distribution

**Rationale**: Shows how to compute gradients for policy optimization. The score function trick allows gradient estimation from samples, enabling policy improvement through gradient ascent.

**Parameter Update**:
```
θt+1 = θt + α∇J(θt)
```
**Terms**:
- `α`: Step size/learning rate

**Rationale**: Standard gradient ascent update rule to improve policy parameters in the direction that increases expected reward.

### REINFORCE
**Purpose**: Monte Carlo policy gradient method.

**Update Rule**:
```
∇θJ(θ) = Eπ[Gt ∇θ ln πθ(At|St)]
```
**Terms**:
- `Gt`: Actual return from time t (Monte Carlo estimate)
- `∇θ ln πθ(At|St)`: Score function for the action taken

**Rationale**: Uses complete episode returns as unbiased estimates of Q-values. Simple but high variance due to Monte Carlo sampling. The return Gt weights the gradient - good episodes increase probability of similar actions.

### Trust Region Policy Optimization (TRPO)
**Purpose**: Constrained policy updates for stable learning.

**Constraint**: KL-divergence between old and new policies stays within trust region.
```
KL(πθ_old || πθ) ≤ δ
```
**Terms**:
- `KL(·||·)`: Kullback-Leibler divergence
- `δ`: Trust region size
- `πθ_old`: Previous policy

**Rationale**: Prevents large policy changes that could be destructive. KL-divergence measures policy similarity, ensuring new policy doesn't deviate too much from current policy.

### Proximal Policy Optimization (PPO)
**Purpose**: Simplified version of TRPO with clipped objective.

**Clipped Objective**:
```
L(θ) = E[min(rt(θ)At, clip(rt(θ), 1-ε, 1+ε)At)]
```
**Terms**:
- `rt(θ) = πθ(at|st)/πθ_old(at|st)`: Probability ratio
- `At`: Advantage estimate
- `ε`: Clipping parameter (typically 0.2)
- `clip(x, min, max)`: Clips x to [min, max]

**Rationale**: Prevents large policy updates by clipping the probability ratio. Simpler than TRPO but achieves similar stability by limiting policy changes.

---

## Actor-Critic Methods

### Basic Actor-Critic
**Components**:
- **Actor**: Updates policy parameters θ using policy gradients
- **Critic**: Estimates value function Qw(s,a) with parameters w

**Actor Update**:
```
θ ← θ + α_actor * Qw(s,a) * ∇θ ln πθ(a|s)
```
**Critic Update**:
```
w ← w + α_critic * (r + γV(s') - Qw(s,a)) * ∇w Qw(s,a)
```

**Rationale**: Combines policy gradients (actor) with value function approximation (critic). Critic reduces variance of policy gradients by providing better baseline than Monte Carlo returns.

### Deep Deterministic Policy Gradient (DDPG)
**Purpose**: Continuous control with deterministic policies.

**Deterministic Policy Gradient**:
```
∇θJ(μθ) = Es~ρβ[∇θ μθ(s) ∇a Q^μ(s,a)|a=μθ(s)]
```
**Terms**:
- `μθ(s)`: Deterministic policy (outputs single action, not distribution)
- `ρβ`: State distribution under behavior policy β
- `∇a Q^μ(s,a)`: Gradient of Q-function with respect to action

**Rationale**: For continuous actions, deterministic policies can be more efficient than stochastic ones. Uses chain rule: gradient of performance with respect to policy parameters equals policy gradient times Q-function gradient.

### Twin Delayed DDPG (TD3)
**Purpose**: Address overestimation in DDPG.

**Key Improvements**:
1. **Twin Critics**: Uses two Q-networks, takes minimum
```
Target = r + γ min(Q1(s',μ(s')), Q2(s',μ(s')))
```
2. **Delayed Updates**: Updates policy less frequently than critics
3. **Target Policy Smoothing**: Adds noise to target actions

**Rationale**: Twin critics reduce overestimation bias. Delayed updates allow critics to improve before policy changes. Target smoothing reduces variance in value estimates.

### Asynchronous Advantage Actor-Critic (A3C)
**Purpose**: Parallel training with multiple agents.

**Advantage Estimate**:
```
A(st,at) = ∑(i=0 to k-1) γ^i * rt+i + γ^k * V(st+k) - V(st)
```
**Terms**:
- `k`: Number of steps for n-step returns
- `V(st)`: State value function
- `A(st,at)`: Advantage function

**Rationale**: Multiple workers collect experience in parallel, reducing correlation. Advantage function (return minus baseline) reduces variance while maintaining unbiased gradient estimates.

### Soft Actor-Critic (SAC)
**Purpose**: Maximum entropy reinforcement learning.

**Objective**:
```
J(π) = ∑t Es~ρπ, a~π[r(st,at) + αH(π(·|st))]
```
**Terms**:
- `H(π(·|st))`: Entropy of policy at state st
- `α`: Temperature parameter balancing reward and entropy

**Rationale**: Maximizes both reward and entropy. High entropy encourages exploration and robustness. Temperature α controls the trade-off between exploitation (high reward) and exploration (high entropy).

### IMPALA (Importance Weighted Actor-Learner Architecture)
**Purpose**: Scalable distributed RL with off-policy correction.

**V-trace Target**:
```
Qret(st,at) = rt + γρt+1[Qret(st+1,at+1) - Q(st+1,at+1)] + γV(st+1)
```
**Terms**:
- `ρt = min(c, πθ(at|st)/μ(at|st))`: Truncated importance sampling ratio
- `c`: Truncation threshold
- `μ`: Behavior policy

**Rationale**: Corrects for off-policy data by weighting updates with importance sampling ratios. Truncation prevents high variance from extreme importance weights.

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
  - Examples: SARSA, A3C, PPO
  - More stable but potentially slower learning
- **Off-Policy**: Training data follows behavior policy β, corrected by importance sampling ratio πθ(a|s)/β(a|s)
  - Examples: Q-learning, DQN, DDPG
  - Better sample efficiency but requires correction methods

### Variance-Bias Trade-off
- **Monte Carlo** (REINFORCE): Unbiased estimates but high variance due to full episode sampling
- **Temporal Difference** (Q-learning): Biased estimates (bootstrap from other estimates) but lower variance
- **Actor-Critic**: Combines both - uses TD learning for critic to reduce variance of policy gradients

### Experience Replay
Experience tuple: `e(s, a, s', r)` represents one interaction with environment.

- **Uniform sampling**: Random selection from replay buffer
  - Simple but treats all experiences equally
- **Prioritized replay**: Sample based on TD-error magnitude
  - Focuses learning on surprising experiences where predictions were wrong

**Rationale**: Replay breaks temporal correlations in sequential data and allows multiple learning updates from single experiences, improving sample efficiency.
