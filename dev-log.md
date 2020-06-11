# 1st Stage: Read & Plan

## Paper Understanding Questions

* Figure 5.
  * is it saying that HIRO is not better than VIME and SNN4HRL in most cases?
  * why the training curve seems weird?

## Fancy Features in Mind

* visitation plot (figure 6.)
* fancy plots
* PyTorch anomaly detection
* w&b

## Brain Storming over Implementation

### Imp Ingredients

* Method Implementation
  * environment playing
  * HRL framework understanding
    * know algorithm well
    * know how to implement with off-policy methods
  * low-level algorithm
    * Q-Learning, DDPG
    * TD3
    * other method to train $\mu^{lo}$?
  * goal-relabeling
    * understand the method (plus math) well
    * understand the algorithm well
    * embed it into naïve HIRO
* Debugging and Tuning
  * debugging
  * use paper's hyper parameters
  * debugging
* Run Experiments

### Time Line

* Environment Preparing (~1 day)
  * read paper & paper code for environment implementation part
  * test and borrow the environment
    * interaction test
    * video saving
    * code base
    * instruction & record
  * commit
* HRL(HIRO) framework understanding (~1 day)
* DDPG understanding (~1 day)
* DDPG Implementation (1~2 day)
* HIRO Implementation (1~2day)
* Debug & Test (1~2 day)
* Util Functions (Debugging Related) (1~2 day)
  * command line interface
  * save & load
  * plotting
* Experiments (1~3 day)
  * Ant Gathering
  * Ant Maze
  * Ant Gathering
  * Ant Fall
* Finish README.md (~1 day)

# 2nd Stage: Coding Prepare

## Environments

* Paper

  > Environments use the MuJoCo simulator with $dt = 0.02$ and frame skip set to 5

  

* Code Base

