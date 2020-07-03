# Value-Based RL: From Q-Learning to HIRO Implementation (developing)

HIRO is a variant of HRL witch is more useful than vanilla HRL. It uses all state-of-the-art techniques to achieve the best result. Understanding and implementing this algorithm from scratch is a little hard to me. I write this document to help clearing my mind. 

HIRO uses TD3 as its RL algorithm. TD3 is a state-of-the-art value-based RL method modified from DDPG. It is applicable to continuous-action tasks, using double DQN, actor-critic, delayed policy update techniques. Implementing TD3 when I only understand Q-learning / DQN settings is hard. Here I try to write down the critical knowledge from Q-learning to TD3, then use them to help me to transform pseudo-code to torch code.

Contents:

* Q-Learning Settings
* DQN settings
* Double DQN
* Actor-Critic Method: Toward continuous tasks

Future contents:

* DQN
  * how to handle sparse reward under TD settings?

* overestimation
  * what is overestimate
  * bias and variance in overestimate
  * why it happens
  * methods of solving
* double Q-Learning
  * understand
  * why
  * code
* sarsa
* dueling DQN
* AC/SAC
* TD3
  * why fill experience memory with random trajectory?
  * why Target Policy Smoothing Regularization works?
* DPG/DDPG
  * why applying DPG is hard? why not direct generate action but generate distribution?
  * what $\pi$ and $\mu$ DPG uses to generate action and critic value?
  * if DDPG is a boring paper that just replace $\pi$ and $\mu$ in DPG with NNs?
    * NNs
    * experience replay
    * target/evaluate NN
* HIRO understanding
  * why HIRO is data efficient when it applies off-policy algorithm

Other References:

* [Deep Reinforcement Learning -1. DDPG原理和算法](https://blog.csdn.net/kenneth_yu/article/details/78478356)
* [深度强化学习之DQN系算法(二) DDPG与TD3算法学习笔记](https://zhuanlan.zhihu.com/p/128477488)
* [深度强化学习系列(16): 从DPG到DDPG算法的原理讲解及tensorflow代码实现](https://blog.csdn.net/gsww404/article/details/80403150)
* [论文阅读-TD3](https://zhuanlan.zhihu.com/p/55307499)

## Q-Learning Settings

### Q-Learning Settings

#### Basic Settings:

The goal is to find a policy that maximizes the long-horizon discounted reward over given environment.

During execution, Q-learning choose the actions that give the maximum expected long-horizon discounted reward at each step. The algorithm evaluate each action at specific sates by a Q function, the training process is basically update these Q functions to their optimal value.

* Goal: $\text{maximize} \, E[\sum_{t=0}^H \gamma^tR(S_t, A_t, S_{t+1} | \pi)]$
* $V_{\pi}(s) = E_{\pi}[U_t | S_t = t] = E[R_{t+1} + \gamma V(s') | S_t = s]$
* $Q(s, a) = E_{\pi}[r_{t+1}, \gamma r_{t+2}, \gamma^2 r_{t+3}, \ldots | A_t = a, S_t = s] = E_{\pi}[G_t | A_t = a, S_t = s]$
* $Q_{\pi}(s, a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V_{\pi}(s')$

When always choose the action that gives the biggest Q value (deterministic policy):  

* $Q_\pi(s, a) = R_s^a + \gamma \sum_{s' \in S} P(s', s, a)[R(s, a, s') + \gamma max_{a'}Q(s',a')]$

#### Update

According to **Bellman Function**:

* $Q^*_\pi(s, a) = R_s^a + \gamma \sum_{s' \in S} P(s', s, a)[R(s, a, s') + \gamma max_{a'}Q^*(s',a')]$

According to **Temporal Difference** update method:

* $V(s) \leftarrow V(s) + \alpha(R_{t+1} + \gamma V(s') - V(s))$
* $Q_\pi(s, a) = Q_{\pi}(s, a) + \alpha[r + \gamma \text{max}_{a'}Q(s', a') - Q(s, a)]$

#### Implementation

check ./review/qlearning.py, the code is straight-forward. 

### Monte-Carlo Update & Temporal-Difference Update

Methods to update policy in RL is different that is DL. in DL we just set a loss function / term and apply backward gradient, gradient descent to minimize it. Since RL setting has time dimension, the updating is more complex than DL. There major updating methods are: **Dynamic Programming**, **Monte-Carlo (MC)** , **Temporal-Difference (TD)** method.

To have a quick knowledge of TD and move forward, we can only remember the formula of TD update:

* $V(s) \leftarrow V(s) + \alpha[R_{t+1} + \gamma V(s') - V(s)]$

The algorithm is also straight forward, actually it was self-contained in Q-Learning algorithm:

<img src="https://img-blog.csdn.net/20180626205846904?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwNjE1OTAz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="td_algorithm" style="zoom: 40%;" />

## DQN settings

### DQN Settings

DQN(Deep Q-Network) is a modification of Q-Learning that solves the problem that Q-learning can not store large amount of Q values. So in DQN we focus on how to implement it, because the problem setting and algorithm is very similar to Q-Learning. 

### Implementation: fundamental DQN code

In DQN, the Q table in Q-Learning was replaced by a NN. 

To refresh memory, Q-Learning updates its Q table via this formula:

* $Q_\pi(s, a) = Q_{\pi}(s, a) + \alpha[r + \gamma \text{max}_{a'}Q(s', a') - Q(s, a)]$

where $r + \gamma \text{max}_{a'}Q(s', a') - Q(s, a)$ is TD difference term, $r + \gamma \text{max}_{a'}Q(s', a')$  is TD target term, also represents the latest Q value estimate. The last $Q(s,a)$ is  current Q value estimation.

####  Fixed Q-targets:

Under above settings, we can use one NN to replace Q table and do the rest just as Q-Learning, But the actual implementation uses two NNs. The purpose of doing this is to leverage bootstrapping runaway bias. See this: [[Why does DQN requrie two different networks?](https://ai.stackexchange.com/questions/6982/why-does-dqn-require-two-different-networks)]

We set one NN as Q_target, which used to calculate TD target term, update after several step interval, and the other one to be Q_evaluate, which represents the last $Q(s, a)$ term, update at each step.

The algorithm first update Q_evaluate at each step follow the updating function but leave Q_target un touched (freeze parameter). After some steps' interval it copy the parameter of Q_evaluate to Q_target to keep Q_target updated. Then repeat till the end of training process. This "technique" is called  "Fixed Q-targets"

We denote these two NNs as:

* Q_evaluate: $Q^-_{\theta^-}(s, a)$
* Q_target: $Q_{\theta}(s, a)$

#### Experience Replay

The actual implementation also applies experience replay technique, i.e., instead of running Q-learning on state/action pairs as they occur during simulation or actual experience, the algorithm stores the data discovered for [state, action, reward, next_state] - typically in a large table. The learning phase is then logically separate from gaining experience, and based on taking random samples from this table.

See this: [[What is "experience replay" and what are its benefits?](https://datascience.stackexchange.com/questions/20535/what-is-experience-replay-and-what-are-its-benefits)]

### Algorithm

<img src="https://morvanzhou.github.io/static/results/reinforcement-learning/4-1-1.jpg" alt="dqn_algorithm" style="zoom:110%;" />

## Double DQN 

### Double DQN Settings

DQN algorithm has two NNs, but Double DQN does not have $2 \times 2 = 4$ NNs. Instead, double DQN uses the two NNs in DQN with different order:

Review the DQN setting:

* Q_evaluate: $Q^-_{\theta^-}(s, a)$
* Q_target: $Q_{\theta}(s, a)$

In DQN, the TD target term is:

* $Y_t^{DQN} \equiv R_{t+1} + \gamma \text{max}_a Q(S_{t+1}, a; \theta_t^-)$

In double DQN:

* $Y_t^{DoubleDQN} \equiv R_{t+1} + \gamma Q(S_{t+1}, \text{max}_a(S_{t+1}, a; \theta_t), \theta_t^-)$

I.E., use Q_evaluate to select the optimal next-step action in Q_target. 

#### Comparation

By setting max Q value to be zero, we can read how Q value overestimate problem from the training curve easily. vanilla DQN's Q value goes beyond zero more, and always above that of double-DQN.

<img src="https://morvanzhou.github.io/static/results/reinforcement-learning/4-5-4.png" alt="double_dqn compare" style="zoom: 67%;" />

 ## Actor-Critic Method: Toward continuous tasks

## DDPG: Value-based RL method solving continuous tasks

### DDPG Settings

DDPG represents deep deterministic policy gradients which combines the concept of deterministic policy gradient and actor-critic framework.

DDPG's frame work is also easy to understand:

* use DPG technique as actor
* use DQN as critic
* joint train actor and critic part by adding their loss term together then perform gradient descent

Techniques in DDPG:

* DPG
* A-C framework
* experience replay
* soft update

To have a quick know ledge for future development, keep in mind:

* DPG generates a deterministic action but not a distribution over action, no need to sample during training and test.
* DPG introduce noise to help exploration during training

#### Algorithm

<img src="https://img-blog.csdn.net/20180522173551120?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dzd3c0MDQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="ddpg_algorithm" style="zoom:80%;" />

## TD3: A more complex DDPG using clipped double Q-learning & delayed policy update

### TD3 Settings:

TD3 represents twin delayed deep deterministic policy gradient, which applies 

* clipped double Q-Learning
* target policy smoothing regularization 
* delayed policy update 
* different noise: use normal distribution to replace OU distribution in DDPG

over DDPG. The main purpose is to leverage overestimate / unstable Q value problem exists in DDPG. 

#### Algorithm

<img src="https://pic4.zhimg.com/80/v2-ebfe3a1163a91140047a514d750d39d3_720w.jpg" alt="img" style="zoom:75%;" />

## HRL: RL method for solving long-horizon tasks

## HIRO: Variant of HRL using off-policy algorithm

## Understanding: from vanilla HRL to HIRO

 ### HRL Basic Idea

#### Off-policy vs. On-policy

* Watch this video: [Reinforcement Learning Class: Off-policy vs On-policy](https://www.youtube.com/watch?v=hlhzvQnXdAA)

Basically, off-policy methods update action policy with rollouts generated by a different xx policy...

Previous HRL methods use on-policy algorithm, .... 

HIRO uses off-policy algorithms to improve ???

But using off-policy algorithm also introduces trouble: The high-level policy will generate a different target for lower-policy after it was (high-level policy) updated. HIRO puts forward a straight forward makeup to solve this problem.

### HIRO: XXXX

### Insights: YYYY