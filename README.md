

# rsl_usr 

This is the codebase for CoRL 2023 paper: [Robust reinforcement learning in continuous control tasks with uncertainty set regularization](https://openreview.net/forum?id=keAPCON4jHC). 

If you use any source codes included in this toolkit in your work, please cite the following paper. The bibtex is listed below:

```bib
    @inproceedings{
    zhang2023robust,
    title={Robust Reinforcement Learning in Continuous Control Tasks with Uncertainty Set Regularization},
    author={Yuan Zhang and Jianhong Wang and Joschka Boedecker},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023},
    url={https://openreview.net/forum?id=keAPCON4jHC}
    }
```

The code implementation largely depends on a Github Repository [Soft Actor-Critic implemtation in Pytorch](https://github.com/denisyarats/pytorch_sac).



## 1. Installation            

Tested on `Ubuntu:20.04` and `Python:3.7`

### Mujoco 2.1.0
Download [Mujoco:2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0) and extract it into `~/.mujoco/mujoco210`

### glew
For conda users, please refer [https://anaconda.org/conda-forge/glew](https://anaconda.org/conda-forge/glew)
```bash
conda install -c conda-forge glew
```

### realwolrdrl_suite
```bash
cd envs/realworldrl_suite
pip install .
```

### other requirements
```bash
pip install -r requirements.txt
```


## 2. Instructions


### 2.1 Training 

To train an SAC agent with `Uncertainty Set Regularizer` introduced in the paper, run `scripts/train.sh`:

```bash
# mujoco 210
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# set up cuda 
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# select exp setting and regularizer setting
override=quadruped_walk
robust_method=l2_adv_param
robust_coef=1e-4
exp_name=adv

# run training on 5 random seeds
cuda_id=0
for seed in 12345 23451 34512 45123 51234; do
    # set up cuda
    export CUDA_VISIBLE_DEVICES=${cuda_id}
    cuda_id=$(($cuda_id+1))
    # run training
    python train.py \
        overrides=${overrides} \
        seed=${seed} \
        agent.params.robust_method=${robust_method} \
        agent.params.robust_coef=${robust_coef} \
        experiment=exp_name &
done
```

* `agent.params.robust_method` is the method to perform additional regularizers. Support methods:
  * `none`: no regularizer on value function, original SAC algorithms
  * `l1_reg`: $l_1$ regularizer on value function 's parameters
  * `l2_reg`: $l_2$ regularizer on value function 's parameters
  * `l1_param`: : $l_1$ uncertainty set regularizer on value function
  * `l2_param`: $l_2$ uncertainty set regularizer on value function
  * `l2_adv_param`: adversarial uncertainty set regularizer on value function

* `agent.params.robust_coef` is the robust magnitude hyperparameter for robust RL
* `experiment` is the special name for this experiment

All experiments are runned on 5 random seeds and saved in `ouputs` folder. One can run tensorboard to monitor training by running `tensorboard --logdir outputs`


### 2.2 Testing 

To test a trained agent under perturbatoins, run `bash scripts/test_robust.sh`

* `perturb_param_list` is the allowed perturbation parameters in the certain environment
* `perturb_min(max)_list` restricts the range of the changed vlaue



## 3. Code Structure

```bash
├── __init__.py
├── agents
│   ├── __init__.py
│   ├── actor.py
│   ├── critic.py
│   └── sac.py: implementation of robust SAC agent, including all robust update methods
├── configs
│   ├── agent
│   ├── overrides: parameters of different tasks
│   └── train.yaml
├── envs
│   ├── realworldrl_suite
│   ├── rwrl_env.py: main entry of RWRL benchmark
│   └── toy_env.py: main entry of toy environment
├── logger.py
├── outputs: folder to save experimental results
├── replay_buffer.py
├── requirements.txt
├── scripts: folder to run training/testing
├── test.py: script to test agents under environmental perturbations
├── train.py: script to train agents
├── utils.py
└── video.py: script to render videos of simulation process
```



