#### Questions**:**

- [x] how to trans form transition tuple sequence to TD3 memory batch?
- [x] clip exploration noise?
- [ ] **what is the maximum goal**
  * look into the code
  * upper bound
- [ ] low-level done judgement







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
  - [ ] **GPU**
  - [ ] <u>**what is the maximum goal**</u>
  - [ ] **wandb video**
- [ ] follow a run
- [ ] run experiment to get good results
- [ ] check imp details with paper / code
  - [ ] networks
  - [ ] hyper-parameters
  - [ ] **log_video function**
- [ ] move to cuda
- [ ] upgrade code efficiency
  * redundant data transformation
- [ ] code cleaning
  * variable name