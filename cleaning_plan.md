#### Questions**:**

- [x] how to trans form transition tuple sequence to TD3 memory batch?
- [x] clip exploration noise?
- [ ] **what is the maximum goal**
  * look into the code
  * upper bound
- [ ] low-level done judgement



#### What To DO

* The code is not running well!
  * If success in 2 million steps, should learn something in 0.25 steps



#### Plan

- [x] add time/state logger
- [x] run experiment
- [x] shower
- [ ] **re-check code again** 
  - [x] util functions
  - [x] TD3
  - [ ] network
  - [x] hierarchy frame
    - [x] intri-reward
    - [x] goal, next_gal, new_goal
    - [x] reward_h
  - [ ] hyper parameters
  - [ ] **noise**, **max_goal**, max_action, terminate condition
- [x] reformat to speedup
  - [x] code clean
  - [ ] **params' unbox**
- [ ] **follow a run**
- [ ] **check env **
  - [ ] on GPU
  - [ ] test AntPush
- [ ] read env codes
- [ ] evaluation code
- [ ] load / save



#### **Final Process:**

- [ ] under stand paper / code base
- [ ] modify code to correct 
  - [x] off-policy correction
  - [x] reward scaling
  - [x] sigma of two noise
  - [x] sigma of goal candidates
  - [ ] check max action(clip)(one of hyper-parameters)
  - [ ] replay buffer size (2e5)
  - [ ] done of high-level
  - [ ] action / goal range
  - [x] **GPU**
  - [ ] <u>**what is the maximum goal**</u>
  - [ ] **wandb video**
  - [ ] load/save
  - [ ] <u>**AntMaze (reward, etc.)**</u>
- [ ] **follow a run**
- [ ] run experiment to get good results
  - [ ] goal weird
- [ ] check imp details with paper / code
  - [ ] networks
  - [ ] hyper-parameters
  - [ ] **log_video function**
- [ ] move to cuda
- [ ] upgrade code efficiency
  * redundant data transformation
- [ ] code cleaning
  * variable name