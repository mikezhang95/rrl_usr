

# mujoco 210
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# set up cuda 
export CUDA_DEVICE_ORDER=PCI_BUS_ID

overrides=walker_stand
# overrides=walker_walk

## non robust
#robust_method=none
#robust_coef=0
 
## l2 reg 
#robust_method=l2_reg
#robust_coef=1e-4
 
## l1 reg 
#robust_method=l1_reg
#robust_coef=1e-4
 
## l2 param
#robust_method=l2_param
#robust_coef=1e-4
 
## l1 param
#robust_method=l1_param
#robust_coef=5e-5

# l2 adv
robust_method=l2_adv_param
robust_coef=1e-4

cuda_id=0
for seed in 12345 23451 34512 45123 51234; do
    # set up cuda
    export CUDA_VISIBLE_DEVICES=${cuda_id}
    cuda_id=$(($cuda_id+1))
    # train
    python train.py \
        overrides=${overrides} \
        seed=${seed} \
        agent.params.robust_method=${robust_method} \
        agent.params.robust_coef=${robust_coef} \
        experiment=adv &
done


