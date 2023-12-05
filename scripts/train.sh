

# mujoco 210
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# set up cuda 
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# select exp setting and regularizer setting
overrides=quadruped_walk
robust_method=l2_adv_param
robust_coef=5e-4
exp_name=adv

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
        experiment=exp_name &
done

